### fake, chat: 360 requests, 4 backends

| Policy | TTFT p50 | TTFT p90 | TTFT p99 | Latency p50 | Prompt tokens from cache | Errors | Requests per backend |
| --- | --- | --- | --- | --- | --- | --- | --- |
| round_robin | 174 ms | 347 ms | 478 ms | 845 ms | 54% | 0 | 90, 90, 90, 90 |
| least_loaded | 174 ms | 354 ms | 479 ms | 846 ms | 53% | 0 | 91, 89, 90, 90 |
| prefix | 91 ms | 95 ms | 238 ms | 761 ms | 79% | 0 | 156, 119, 77, 8 |
