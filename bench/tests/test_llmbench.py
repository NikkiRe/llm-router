from llmbench import client, report, workload


def test_documents_are_deterministic_and_share_prefixes():
    a = workload.documents(n_docs=3, doc_words=50, n_requests=20, rate=2, zipf_s=1, seed=7)
    b = workload.documents(n_docs=3, doc_words=50, n_requests=20, rate=2, zipf_s=1, seed=7)
    assert a == b
    assert all(x.at < y.at for x, y in zip(a, a[1:], strict=False))
    same_doc = [r for r in a if r.group == a[0].group]
    assert len({r.messages[0]["content"] for r in same_doc}) == 1


def test_zipf_makes_first_document_hottest():
    reqs = workload.documents(n_docs=5, doc_words=10, n_requests=500, rate=10, zipf_s=1.2, seed=1)
    counts = {g: sum(r.group == g for r in reqs) for g in {r.group for r in reqs}}
    assert max(counts, key=counts.get) == "doc0"


def test_chat_turns_extend_history():
    reqs = workload.chat(
        n_sessions=2, turns=3, system_words=20, turn_words=10, rate=1, think=1, seed=3
    )
    session = [r for r in reqs if r.group == "session0"]
    assert len(session) == 3
    for prev, nxt in zip(session, session[1:], strict=False):
        assert nxt.messages[: len(prev.messages)] == prev.messages


def test_has_token_ignores_role_only_event():
    assert not client.has_token({"choices": [{"delta": {"role": "assistant"}}]})
    assert client.has_token({"choices": [{"delta": {"content": "Hi"}}]})
    assert client.has_token({"choices": [{"delta": {"reasoning_content": "Hmm"}}]})


def test_cache_counts_reads_openai_and_llamacpp():
    usage = {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 80}}
    assert client.cache_counts(usage, {}) == (100, 80)
    assert client.cache_counts({}, {"cache_n": 70, "prompt_n": 30}) == (100, 70)


def test_percentile_interpolates():
    assert report.percentile([1, 2, 3, 4], 50) == 2.5
    assert report.percentile([5], 99) == 5
    assert report.percentile([], 50) is None
