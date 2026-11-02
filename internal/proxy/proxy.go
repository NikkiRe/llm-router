// Package proxy serves the OpenAI-compatible API and forwards each request to
// the backend the pool picks.
package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"log/slog"
	"math"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/NikkiRe/llm-router/internal/balancer"
	"github.com/NikkiRe/llm-router/internal/limit"
	"github.com/NikkiRe/llm-router/internal/metrics"
	"github.com/NikkiRe/llm-router/internal/openai"
	"github.com/NikkiRe/llm-router/internal/prefix"
)

// Config holds the proxy settings.
type Config struct {
	// ChunkSize is the prefix chunk length in bytes, see prefix.Hashes.
	ChunkSize int
	// QueueTimeout bounds how long a request waits for a free backend slot.
	QueueTimeout time.Duration
	// Retries is how many other backends to try when one fails before
	// sending any response.
	Retries int
	// MaxBodyBytes caps the request body.
	MaxBodyBytes int64
	// APIKeys, if not empty, are the only keys accepted.
	APIKeys map[string]bool
}

// Proxy is the router's public HTTP handler.
type Proxy struct {
	cfg     Config
	pool    *balancer.Pool
	limiter *limit.Limiter
	metrics *metrics.Metrics
	client  *http.Client
	log     *slog.Logger
	mux     *http.ServeMux
}

// New returns a proxy that sends requests through transport.
func New(cfg Config, pool *balancer.Pool, limiter *limit.Limiter, m *metrics.Metrics,
	transport http.RoundTripper, log *slog.Logger) *Proxy {
	p := &Proxy{
		cfg:     cfg,
		pool:    pool,
		limiter: limiter,
		metrics: m,
		// No client timeout: a generation may stream for minutes. Clients
		// end it by disconnecting, which cancels the upstream request.
		client: &http.Client{Transport: transport},
		log:    log,
		mux:    http.NewServeMux(),
	}
	p.mux.HandleFunc("POST /v1/chat/completions", p.completions)
	p.mux.HandleFunc("POST /v1/completions", p.completions)
	p.mux.HandleFunc("GET /v1/models", p.models)
	return p
}

func (p *Proxy) ServeHTTP(w http.ResponseWriter, r *http.Request) { p.mux.ServeHTTP(w, r) }

func (p *Proxy) completions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	key := apiKey(r)
	if len(p.cfg.APIKeys) > 0 && !p.cfg.APIKeys[key] {
		p.reject(w, http.StatusUnauthorized, "invalid_api_key", "invalid API key")
		return
	}

	release, wait, err := p.limiter.Acquire(key)
	switch {
	case errors.Is(err, limit.ErrRateLimited):
		w.Header().Set("Retry-After", retryAfter(wait))
		p.reject(w, http.StatusTooManyRequests, "rate_limited", err.Error())
		return
	case errors.Is(err, limit.ErrTooManyConcurrent):
		p.reject(w, http.StatusTooManyRequests, "too_many_concurrent", err.Error())
		return
	}
	defer release()

	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, p.cfg.MaxBodyBytes))
	if err != nil {
		var tooBig *http.MaxBytesError
		if errors.As(err, &tooBig) {
			p.reject(w, http.StatusRequestEntityTooLarge, "body_too_large", "request body too large")
		} else {
			p.reject(w, http.StatusBadRequest, "bad_request", "cannot read request body")
		}
		return
	}
	req, err := openai.Parse(r.URL.Path, body)
	if err != nil {
		p.reject(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	route := balancer.Request{Hashes: prefix.Hashes(req.Prompt, p.cfg.ChunkSize)}

	for attempt := 0; ; attempt++ {
		queueCtx, cancel := context.WithTimeout(r.Context(), p.cfg.QueueTimeout)
		lease, err := p.pool.Acquire(queueCtx, route)
		cancel()
		if err != nil {
			switch {
			case r.Context().Err() != nil:
				p.metrics.Rejected.WithLabelValues("client_gone").Inc()
			case errors.Is(err, balancer.ErrNoBackend):
				p.reject(w, http.StatusServiceUnavailable, "no_backend", "no healthy backend")
			default:
				w.Header().Set("Retry-After", "1")
				p.reject(w, http.StatusServiceUnavailable, "queue_timeout", "all backends are busy")
			}
			return
		}
		p.metrics.Decisions.WithLabelValues(lease.Reason).Inc()
		if n := len(route.Hashes); n > 0 {
			p.metrics.MatchRatio.Observe(float64(lease.Matched) / float64(n))
		}

		resp, err := p.send(r, lease.Backend, body)
		if err == nil && !retryable(resp.StatusCode) {
			p.pool.Observe(lease.Backend, true)
			p.relay(w, resp, lease.Backend, start)
			lease.Release()
			return
		}

		// Nothing has been written to the client yet, so another backend
		// can still take the request.
		if r.Context().Err() != nil {
			// The client left; that says nothing about the backend.
			closeBody(resp)
			lease.Release()
			p.metrics.Rejected.WithLabelValues("client_gone").Inc()
			return
		}
		p.pool.Observe(lease.Backend, false)
		if attempt >= p.cfg.Retries {
			if err == nil {
				p.relay(w, resp, lease.Backend, start) // pass the backend's error on
			} else {
				p.log.Warn("backend request failed", "backend", lease.Backend.Name, "err", err)
				p.metrics.Requests.WithLabelValues(lease.Backend.Name, "502").Inc()
				openai.WriteError(w, http.StatusBadGateway, "bad_gateway", "backend request failed")
			}
			lease.Release()
			return
		}
		p.log.Info("retrying on another backend", "backend", lease.Backend.Name, "err", err, "status", status(resp))
		closeBody(resp)
		lease.Release()
		route.Exclude = append(route.Exclude, lease.Backend)
		p.metrics.Retries.Inc()
	}
}

// models answers from any healthy backend; the model list is the same
// across a pool.
func (p *Proxy) models(w http.ResponseWriter, r *http.Request) {
	for _, b := range p.pool.Backends() {
		if !b.Healthy() {
			continue
		}
		resp, err := p.send(r, b, nil)
		if err != nil {
			continue
		}
		defer resp.Body.Close()
		copyHeader(w.Header(), resp.Header)
		w.WriteHeader(resp.StatusCode)
		_, _ = io.Copy(w, resp.Body)
		return
	}
	p.reject(w, http.StatusServiceUnavailable, "no_backend", "no healthy backend")
}

func (p *Proxy) send(r *http.Request, b *balancer.Backend, body []byte) (*http.Response, error) {
	u := *b.URL
	u.Path = strings.TrimRight(b.URL.Path, "/") + r.URL.Path
	u.RawQuery = r.URL.RawQuery
	out, err := http.NewRequestWithContext(r.Context(), r.Method, u.String(), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	copyHeader(out.Header, r.Header)
	// The client's key is for the router; the backend may have its own.
	out.Header.Del("Authorization")
	out.Header.Del("Cookie")
	// Let the transport negotiate compression and decompress for us; a
	// forwarded Accept-Encoding would hand us gzip bytes to parse as SSE.
	out.Header.Del("Accept-Encoding")
	if b.APIKey != "" {
		out.Header.Set("Authorization", "Bearer "+b.APIKey)
	}
	return p.client.Do(out)
}

func (p *Proxy) relay(w http.ResponseWriter, resp *http.Response, b *balancer.Backend, start time.Time) {
	defer resp.Body.Close()
	copyHeader(w.Header(), resp.Header)
	w.Header().Set("X-Router-Backend", b.Name)
	w.WriteHeader(resp.StatusCode)

	var usage *openai.Usage
	if strings.HasPrefix(resp.Header.Get("Content-Type"), "text/event-stream") {
		usage = p.relayStream(w, resp.Body, b, start)
	} else {
		usage = relayBody(w, resp.Body)
	}
	if usage != nil {
		p.metrics.Tokens.WithLabelValues(b.Name, "prompt").Add(float64(usage.PromptTokens))
		p.metrics.Tokens.WithLabelValues(b.Name, "completion").Add(float64(usage.CompletionTokens))
		if d := usage.PromptTokensDetails; d != nil {
			p.metrics.Tokens.WithLabelValues(b.Name, "cached").Add(float64(d.CachedTokens))
		}
	}
	p.metrics.Requests.WithLabelValues(b.Name, strconv.Itoa(resp.StatusCode)).Inc()
	p.metrics.Duration.WithLabelValues(b.Name).Observe(time.Since(start).Seconds())
}

var (
	dataPrefix = []byte("data:")
	doneEvent  = []byte("data: [DONE]")
	usageField = []byte(`"usage"`)
)

// relayStream copies server-sent events line by line and flushes at every
// event boundary, so tokens reach the client as soon as the backend emits
// them. It returns the last usage the stream reported.
func (p *Proxy) relayStream(w http.ResponseWriter, body io.Reader, b *balancer.Backend, start time.Time) *openai.Usage {
	rc := http.NewResponseController(w)
	br := bufio.NewReaderSize(body, 64<<10)
	first := true
	var usage *openai.Usage
	for {
		line, err := br.ReadSlice('\n')
		if len(line) > 0 {
			if bytes.HasPrefix(line, dataPrefix) && !bytes.HasPrefix(line, doneEvent) {
				if first && hasToken(bytes.TrimSpace(line[len(dataPrefix):])) {
					p.metrics.TTFT.WithLabelValues(b.Name).Observe(time.Since(start).Seconds())
					first = false
				}
				p.metrics.Chunks.WithLabelValues(b.Name).Inc()
				if bytes.Contains(line, usageField) {
					if u := parseUsage(bytes.TrimSpace(line[len(dataPrefix):])); u != nil {
						usage = u
					}
				}
			}
			if _, werr := w.Write(line); werr != nil {
				return usage // client disconnected; closing the body cancels generation
			}
			if len(bytes.TrimSpace(line)) == 0 {
				_ = rc.Flush()
			}
		}
		if err != nil {
			if errors.Is(err, bufio.ErrBufferFull) {
				continue // a line longer than the buffer: keep copying it
			}
			_ = rc.Flush()
			return usage
		}
	}
}

// hasToken reports whether a stream event carries generated text. Servers
// may open a stream with an event that only announces the assistant role,
// and that event must not count as the first token. Only the first few
// events of a stream are parsed.
func hasToken(data []byte) bool {
	var ev struct {
		Choices []struct {
			Text  string `json:"text"`
			Delta struct {
				Content          string `json:"content"`
				ReasoningContent string `json:"reasoning_content"`
			} `json:"delta"`
		} `json:"choices"`
	}
	if json.Unmarshal(data, &ev) != nil {
		return false
	}
	for _, c := range ev.Choices {
		if c.Text != "" || c.Delta.Content != "" || c.Delta.ReasoningContent != "" {
			return true
		}
	}
	return false
}

func relayBody(w http.ResponseWriter, body io.Reader) *openai.Usage {
	data, err := io.ReadAll(body)
	if len(data) > 0 {
		_, _ = w.Write(data)
	}
	if err != nil {
		return nil
	}
	return parseUsage(data)
}

func parseUsage(data []byte) *openai.Usage {
	var v struct {
		Usage *openai.Usage `json:"usage"`
	}
	if json.Unmarshal(data, &v) != nil {
		return nil
	}
	return v.Usage
}

func (p *Proxy) reject(w http.ResponseWriter, code int, reason, msg string) {
	p.metrics.Rejected.WithLabelValues(reason).Inc()
	openai.WriteError(w, code, reason, msg)
}

func apiKey(r *http.Request) string {
	if k, ok := strings.CutPrefix(r.Header.Get("Authorization"), "Bearer "); ok && k != "" {
		return k
	}
	return "anonymous"
}

func retryAfter(d time.Duration) string {
	return strconv.Itoa(max(1, int(math.Ceil(d.Seconds()))))
}

// retryable reports statuses that mean "this backend cannot serve now":
// a llama.cpp server answers 503 while it loads the model.
func retryable(code int) bool {
	return code == http.StatusBadGateway || code == http.StatusServiceUnavailable || code == http.StatusGatewayTimeout
}

func status(resp *http.Response) int {
	if resp == nil {
		return 0
	}
	return resp.StatusCode
}

func closeBody(resp *http.Response) {
	if resp != nil {
		_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, 64<<10))
		resp.Body.Close()
	}
}

// Hop-by-hop headers belong to one connection and are not forwarded.
// Content-Length is dropped as well: the server sets its own.
var hopByHop = map[string]bool{
	"Connection": true, "Keep-Alive": true, "Proxy-Authenticate": true,
	"Proxy-Authorization": true, "Te": true, "Trailer": true,
	"Transfer-Encoding": true, "Upgrade": true, "Content-Length": true,
}

func copyHeader(dst, src http.Header) {
	for k, vv := range src {
		if hopByHop[k] {
			continue
		}
		dst[k] = append([]string(nil), vv...)
	}
}
