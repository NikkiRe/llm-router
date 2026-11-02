"""Sends a workload to the router and records what each request experienced."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass

import httpx

from .workload import Request


@dataclass
class Result:
    group: str
    scheduled: float  # planned send time, seconds after the start
    ttft: float | None  # seconds from send to the first generated token
    latency: float | None  # seconds from send to the end of the stream
    status: int
    backend: str
    prompt_tokens: int
    cached_tokens: int
    error: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self))


def has_token(event: dict) -> bool:
    for c in event.get("choices") or []:
        delta = c.get("delta") or {}
        if c.get("text") or delta.get("content") or delta.get("reasoning_content"):
            return True
    return False


def cache_counts(usage: dict, timings: dict) -> tuple[int, int]:
    """Prompt and cached tokens. OpenAI-style servers report them in usage;
    llama.cpp also reports cache_n (reused) and prompt_n (computed) in
    timings."""
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if timings:
        if cached is None:
            cached = timings.get("cache_n", 0)
        if not prompt:
            prompt = timings.get("cache_n", 0) + timings.get("prompt_n", 0)
    return prompt, cached or 0


async def send(client: httpx.AsyncClient, url: str, req: Request, max_tokens: int) -> Result:
    body = {
        "model": "default",
        "messages": req.messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    result = Result(req.group, req.at, None, None, 0, "", 0, 0)
    try:
        async with client.stream("POST", url + "/v1/chat/completions", json=body) as resp:
            result.status = resp.status_code
            result.backend = resp.headers.get("x-router-backend", "")
            if resp.status_code != 200:
                result.error = (await resp.aread()).decode(errors="replace")[:200]
                return result
            usage: dict = {}
            timings: dict = {}
            async for line in resp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if result.ttft is None and has_token(event):
                    result.ttft = time.perf_counter() - start
                usage = event.get("usage") or usage
                timings = event.get("timings") or timings
            result.latency = time.perf_counter() - start
            result.prompt_tokens, result.cached_tokens = cache_counts(usage, timings)
    except httpx.HTTPError as e:
        result.error = f"{type(e).__name__}: {e}"
    return result


async def run(url: str, workload: list[Request], max_tokens: int, timeout: float) -> list[Result]:
    """Sends every request at its planned time (open loop): a slow router does
    not slow the arrivals down, it builds a queue, as in production."""
    limits = httpx.Limits(max_connections=None, max_keepalive_connections=256)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        t0 = time.perf_counter()

        async def one(req: Request) -> Result:
            delay = req.at - (time.perf_counter() - t0)
            if delay > 0:
                await asyncio.sleep(delay)
            return await send(client, url, req, max_tokens)

        return await asyncio.gather(*(one(r) for r in workload))
