"""Runs the same workload through the router once per routing policy.

Backends are restarted before every policy, so no policy inherits a cache
warmed by the previous one.

    uv run python -m llmbench.run --backend fake --scenario docs --out results/fake-docs
    uv run python -m llmbench.run --backend llamacpp --model ~/models/qwen.gguf \\
        --scenario docs --out results/llamacpp-docs
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

from . import client, report, workload

REPO = Path(__file__).resolve().parents[2]
BIN = REPO / "bin"
ROUTER_PORT, ADMIN_PORT, FIRST_BACKEND_PORT = 18080, 19090, 18101
POLICIES = ("round_robin", "least_loaded", "prefix")


def build() -> None:
    subprocess.run(["go", "build", "-o", str(BIN) + "/", "./cmd/..."], cwd=REPO, check=True)


def wait_ok(url: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with contextlib.suppress(httpx.HTTPError):
            if httpx.get(url, timeout=2).status_code == 200:
                return
        time.sleep(0.25)
    raise TimeoutError(f"{url} did not become ready in {timeout:.0f}s")


@contextlib.contextmanager
def process(cmd: list[str], log: Path):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as f:
        p = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        try:
            yield p
        finally:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait()


def backend_commands(args) -> list[list[str]]:
    cmds = []
    for i in range(args.instances):
        port = FIRST_BACKEND_PORT + i
        if args.backend == "fake":
            cmds.append(
                [
                    str(BIN / "fakellm"),
                    "-addr",
                    f"127.0.0.1:{port}",
                    "-name",
                    f"fake{i}",
                    "-slots",
                    str(args.slots),
                    "-prefill-tps",
                    str(args.fake_prefill_tps),
                    "-decode-tps",
                    str(args.fake_decode_tps),
                    "-cache-tokens",
                    str(args.fake_cache_tokens),
                ]
            )
        else:
            cmds.append(
                [
                    "llama-server",
                    "-m",
                    str(Path(args.model).expanduser()),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "-np",
                    str(args.slots),
                    "-c",
                    str(args.ctx),
                    "-ngl",
                    "99",
                    "--cache-ram",
                    str(args.cache_ram),
                    "--no-webui",
                ]
            )
    return cmds


def router_config(args, path: Path) -> None:
    backends = "\n".join(
        f"  - {{name: {args.backend}{i}, url: 'http://127.0.0.1:{FIRST_BACKEND_PORT + i}', "
        f"max_inflight: {args.slots}}}"
        for i in range(args.instances)
    )
    path.write_text(
        f"listen: '127.0.0.1:{ROUTER_PORT}'\n"
        f"admin_listen: '127.0.0.1:{ADMIN_PORT}'\n"
        f"backends:\n{backends}\n"
        f"health: {{interval: 1s, timeout: 2s}}\n"
        f"prefix: {{chunk_size: {args.chunk_size}, min_match_ratio: 0.5, "
        f"load_slack: {args.slack}}}\n"
        f"queue_timeout: 120s\n"
        f"response_header_timeout: 10m\n"
    )


def make_workload(args) -> list[workload.Request]:
    if args.scenario == "docs":
        return workload.documents(
            n_docs=args.docs,
            doc_words=args.doc_words,
            n_requests=args.requests,
            rate=args.rate,
            zipf_s=args.zipf,
            seed=args.seed,
        )
    return workload.chat(
        n_sessions=args.sessions,
        turns=args.turns,
        system_words=args.doc_words,
        turn_words=args.turn_words,
        rate=args.rate,
        think=args.think,
        seed=args.seed,
    )


def run_policy(args, policy: str, reqs: list[workload.Request], out: Path) -> dict:
    raw = out / "raw" / policy
    with contextlib.ExitStack() as stack:
        for i, cmd in enumerate(backend_commands(args)):
            stack.enter_context(process(cmd, raw / f"backend{i}.log"))
        for i in range(args.instances):
            wait_ok(f"http://127.0.0.1:{FIRST_BACKEND_PORT + i}/health", args.ready_timeout)
        stack.enter_context(
            process(
                [str(BIN / "llmrouter"), "-config", str(out / "router.yaml"), "-policy", policy],
                raw / "router.log",
            )
        )
        wait_ok(f"http://127.0.0.1:{ADMIN_PORT}/healthz", 10)

        print(f"[{policy}] {len(reqs)} requests over {reqs[-1].at:.0f}s", flush=True)
        t0 = time.perf_counter()
        results = asyncio.run(
            client.run(f"http://127.0.0.1:{ROUTER_PORT}", reqs, args.max_tokens, args.timeout)
        )
        wall = time.perf_counter() - t0
        (raw / "metrics.txt").write_text(httpx.get(f"http://127.0.0.1:{ADMIN_PORT}/metrics").text)

    with (raw / "results.jsonl").open("w") as f:
        for r in results:
            f.write(r.to_json() + "\n")
    s = report.summarize(results, wall)
    print(f"[{policy}] " + report.one_line(s), flush=True)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--backend", choices=("fake", "llamacpp"), default="fake")
    ap.add_argument("--model", help="GGUF model for --backend llamacpp")
    ap.add_argument("--scenario", choices=("docs", "chat"), default="docs")
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-build", action="store_true")
    ap.add_argument("--instances", type=int, default=4)
    ap.add_argument("--slots", type=int, default=4, help="parallel requests per backend")
    ap.add_argument("--requests", type=int, default=400)
    ap.add_argument(
        "--rate", type=float, default=4.0, help="requests (docs) or sessions (chat) per second"
    )
    ap.add_argument("--docs", type=int, default=16)
    ap.add_argument("--doc-words", type=int, default=1500, help="document or system prompt length")
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--sessions", type=int, default=60)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--turn-words", type=int, default=120)
    ap.add_argument("--think", type=float, default=4.0, help="seconds between turns of a chat")
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=256)
    ap.add_argument("--slack", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--ready-timeout", type=float, default=180)
    ap.add_argument("--fake-prefill-tps", type=float, default=4000)
    ap.add_argument("--fake-decode-tps", type=float, default=50)
    ap.add_argument("--fake-cache-tokens", type=int, default=32768)
    ap.add_argument("--ctx", type=int, default=8192, help="llama.cpp context, shared by slots")
    ap.add_argument("--cache-ram", type=int, default=1024, help="llama.cpp host prompt cache, MiB")
    args = ap.parse_args()
    if args.backend == "llamacpp" and not args.model:
        ap.error("--backend llamacpp needs --model")

    if not args.no_build:
        build()
    args.out.mkdir(parents=True, exist_ok=True)
    router_config(args, args.out / "router.yaml")
    reqs = make_workload(args)

    summaries = {}
    for policy in args.policies.split(","):
        summaries[policy] = run_policy(args, policy, reqs, args.out)

    setup = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    if args.model:
        setup["model"] = Path(args.model).name  # no local paths in published results
    (args.out / "summary.json").write_text(
        json.dumps({"setup": setup, "policies": summaries}, indent=2) + "\n"
    )
    title = f"{args.backend}, {args.scenario}: {len(reqs)} requests, {args.instances} backends"
    (args.out / "summary.md").write_text(report.markdown(summaries, title))
    report.chart(summaries, args.out / "ttft.png", title)
    print(f"wrote {args.out}/summary.md and ttft.png")


if __name__ == "__main__":
    sys.exit(main())
