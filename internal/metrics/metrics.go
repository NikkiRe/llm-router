// Package metrics defines the router's Prometheus metrics.
package metrics

import (
	"github.com/prometheus/client_golang/prometheus"

	"github.com/NikkiRe/llm-router/internal/balancer"
)

// Metrics groups every collector the router updates.
type Metrics struct {
	Requests   *prometheus.CounterVec
	Duration   *prometheus.HistogramVec
	TTFT       *prometheus.HistogramVec
	Chunks     *prometheus.CounterVec
	Tokens     *prometheus.CounterVec
	Decisions  *prometheus.CounterVec
	MatchRatio prometheus.Histogram
	Rejected   *prometheus.CounterVec
	Retries    prometheus.Counter
}

// latencyBuckets span cached prompts on a GPU (milliseconds) to long CPU
// prefills (tens of seconds).
var latencyBuckets = []float64{.01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10, 25, 60}

// New registers the metrics, plus per-backend gauges that read the pool.
func New(reg prometheus.Registerer, pool *balancer.Pool) *Metrics {
	m := &Metrics{
		Requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llmrouter_requests_total",
			Help: "Proxied requests by backend and HTTP status.",
		}, []string{"backend", "code"}),
		Duration: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "llmrouter_request_duration_seconds",
			Help:    "Time from receiving a request to the end of its response.",
			Buckets: latencyBuckets,
		}, []string{"backend"}),
		TTFT: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name:    "llmrouter_time_to_first_token_seconds",
			Help:    "Time from receiving a streaming request to its first data event.",
			Buckets: latencyBuckets,
		}, []string{"backend"}),
		Chunks: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llmrouter_stream_chunks_total",
			Help: "Data events relayed on streaming responses, roughly one per token.",
		}, []string{"backend"}),
		Tokens: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llmrouter_tokens_total",
			Help: "Tokens reported by backends in usage, by kind: prompt, cached or completion.",
		}, []string{"backend", "kind"}),
		Decisions: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llmrouter_routing_decisions_total",
			Help: "Routing decisions by reason: prefix, least_loaded or round_robin.",
		}, []string{"reason"}),
		MatchRatio: prometheus.NewHistogram(prometheus.HistogramOpts{
			Name:    "llmrouter_prefix_match_ratio",
			Help:    "Share of the prompt the chosen backend was believed to have cached.",
			Buckets: prometheus.LinearBuckets(0, 0.1, 11),
		}),
		Rejected: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "llmrouter_rejected_total",
			Help: "Requests refused before reaching a backend, by reason.",
		}, []string{"reason"}),
		Retries: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "llmrouter_retries_total",
			Help: "Requests sent to another backend after the first one failed.",
		}),
	}
	reg.MustRegister(m.Requests, m.Duration, m.TTFT, m.Chunks, m.Tokens,
		m.Decisions, m.MatchRatio, m.Rejected, m.Retries)

	for _, b := range pool.Backends() {
		labels := prometheus.Labels{"backend": b.Name}
		reg.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Name:        "llmrouter_backend_inflight",
			Help:        "Requests running on the backend.",
			ConstLabels: labels,
		}, func() float64 { return float64(b.Inflight()) }))
		reg.MustRegister(prometheus.NewGaugeFunc(prometheus.GaugeOpts{
			Name:        "llmrouter_backend_healthy",
			Help:        "1 if the backend receives traffic, 0 if it is ejected.",
			ConstLabels: labels,
		}, func() float64 {
			if b.Healthy() {
				return 1
			}
			return 0
		}))
	}
	return m
}
