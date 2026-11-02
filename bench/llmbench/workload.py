"""Request streams for the benchmark.

Everything is generated from a seed, so every routing policy sees exactly the
same requests at the same moments.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

WORDS = (
    "cache prefill decode token latency router backend queue model prompt batch kernel memory "
    "throughput request stream document question answer context window shard replica gpu "
    "scheduler slot eviction policy affinity load balance health retry timeout metric"
).split()


@dataclass(frozen=True)
class Request:
    at: float  # seconds after the start of the run
    messages: list[dict]
    group: str  # the document or chat session the request belongs to


def text(rng: random.Random, n_words: int) -> str:
    sentences = []
    while n_words > 0:
        k = min(n_words, rng.randint(8, 16))
        words = [rng.choice(WORDS) for _ in range(k)]
        sentences.append(" ".join(words).capitalize() + ".")
        n_words -= k
    return " ".join(sentences)


def zipf_weights(n: int, s: float) -> list[float]:
    """Popularity of n items: item k is (k+1)**s times rarer than the first."""
    return [1 / (k + 1) ** s for k in range(n)]


def documents(
    *,
    n_docs: int,
    doc_words: int,
    n_requests: int,
    rate: float,
    zipf_s: float,
    seed: int,
) -> list[Request]:
    """Questions about shared documents, like RAG over a few large sources or
    many tenants with long system prompts. Arrivals are Poisson at `rate` per
    second; documents are picked with Zipf popularity, so some are hot."""
    rng = random.Random(seed)
    docs = [f"Document {d}.\n{text(rng, doc_words)}" for d in range(n_docs)]
    weights = zipf_weights(n_docs, zipf_s)
    t = 0.0
    out = []
    for i in range(n_requests):
        t += rng.expovariate(rate)
        d = rng.choices(range(n_docs), weights)[0]
        question = f"Question {i}: what does the document say about {rng.choice(WORDS)}?"
        out.append(
            Request(
                at=t,
                group=f"doc{d}",
                messages=[
                    {"role": "system", "content": "Answer from the document.\n\n" + docs[d]},
                    {"role": "user", "content": question},
                ],
            )
        )
    return out


def chat(
    *,
    n_sessions: int,
    turns: int,
    system_words: int,
    turn_words: int,
    rate: float,
    think: float,
    seed: int,
) -> list[Request]:
    """Multi-turn chats: each turn resends the whole history, so it shares its
    prefix with the previous turn of the same session. Sessions start as a
    Poisson process; turns follow every `think` seconds. Assistant replies are
    synthetic, so the shared prefix ends at the previous user message plus
    the reply text."""
    rng = random.Random(seed)
    system = {"role": "system", "content": text(rng, system_words)}
    out = []
    start = 0.0
    for s in range(n_sessions):
        start += rng.expovariate(rate)
        history = [system]
        for k in range(turns):
            history = history + [{"role": "user", "content": text(rng, turn_words)}]
            out.append(Request(at=start + k * think, group=f"session{s}", messages=history))
            history = history + [{"role": "assistant", "content": text(rng, turn_words // 2)}]
    out.sort(key=lambda r: r.at)
    return out
