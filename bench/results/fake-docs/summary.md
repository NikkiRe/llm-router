### fake, docs: 400 requests, 4 backends

| Policy | TTFT p50 | TTFT p90 | TTFT p99 | Latency p50 | Prompt tokens from cache | Errors | Requests per backend |
| --- | --- | --- | --- | --- | --- | --- | --- |
| round_robin | 11 ms | 700 ms | 954 ms | 684 ms | 78% | 0 | 100, 100, 100, 100 |
| least_loaded | 11 ms | 700 ms | 954 ms | 684 ms | 77% | 0 | 101, 98, 100, 101 |
| prefix | 11 ms | 14 ms | 704 ms | 681 ms | 93% | 0 | 85, 138, 75, 102 |
