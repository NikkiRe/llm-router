### llamacpp, docs: 80 requests, 2 backends

| Policy | TTFT p50 | TTFT p90 | TTFT p99 | Latency p50 | Prompt tokens from cache | Errors | Requests per backend |
| --- | --- | --- | --- | --- | --- | --- | --- |
| round_robin | 847 ms | 5.36 s | 8.39 s | 1.81 s | 64% | 0 | 41, 39 |
| least_loaded | 822 ms | 5.34 s | 8.35 s | 1.85 s | 64% | 0 | 41, 39 |
| prefix | 229 ms | 3.72 s | 8.38 s | 713 ms | 75% | 0 | 44, 36 |
