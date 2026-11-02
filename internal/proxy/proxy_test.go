package proxy

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/testutil"

	"github.com/NikkiRe/llm-router/internal/balancer"
	"github.com/NikkiRe/llm-router/internal/fakellm"
	"github.com/NikkiRe/llm-router/internal/limit"
	"github.com/NikkiRe/llm-router/internal/metrics"
)

type options struct {
	policy       string
	maxInflight  int
	retries      int
	fall         int
	queueTimeout time.Duration
	limiter      *limit.Limiter
	apiKeys      map[string]bool
	backendKey   string
}

type env struct {
	url      string
	backends []*balancer.Backend
	metrics  *metrics.Metrics
}

// newEnv starts one backend per handler and a router in front of them. A nil
// handler is a backend that refuses connections.
func newEnv(t *testing.T, handlers []http.Handler, o options) *env {
	t.Helper()
	if o.policy == "" {
		o.policy = "prefix"
	}
	if o.queueTimeout == 0 {
		o.queueTimeout = 5 * time.Second
	}
	if o.limiter == nil {
		o.limiter = limit.New(0, 0, 0)
	}
	var backends []*balancer.Backend
	for i, h := range handlers {
		srv := httptest.NewServer(h)
		if h == nil {
			srv.Close()
		} else {
			t.Cleanup(srv.Close)
		}
		u, _ := url.Parse(srv.URL)
		b := balancer.NewBackend(fmt.Sprintf("b%d", i), u, o.maxInflight, 4096)
		b.APIKey = o.backendKey
		backends = append(backends, b)
	}
	policy, err := balancer.NewPolicy(o.policy, 0.5, 4)
	if err != nil {
		t.Fatal(err)
	}
	pool := balancer.NewPool(backends, policy, balancer.Thresholds{Fall: max(o.fall, 1), Rise: 1})
	m := metrics.New(prometheus.NewRegistry(), pool)
	p := New(Config{
		ChunkSize:    64,
		QueueTimeout: o.queueTimeout,
		Retries:      o.retries,
		MaxBodyBytes: 1 << 20,
		APIKeys:      o.apiKeys,
	}, pool, o.limiter, m, http.DefaultTransport, slog.New(slog.DiscardHandler))
	srv := httptest.NewServer(p)
	t.Cleanup(srv.Close)
	return &env{url: srv.URL, backends: backends, metrics: m}
}

func fastFake(name string) *fakellm.Server {
	cfg := fakellm.DefaultConfig(name)
	cfg.PrefillTPS = 1e6
	cfg.DecodeTPS = 1e4
	cfg.MaxTokens = 5
	return fakellm.New(cfg)
}

func chatBody(system, user string, stream bool) string {
	b, _ := json.Marshal(map[string]any{
		"model":          "fake",
		"stream":         stream,
		"stream_options": map[string]any{"include_usage": stream},
		"messages": []any{
			map[string]any{"role": "system", "content": system},
			map[string]any{"role": "user", "content": user},
		},
	})
	return string(b)
}

func post(t *testing.T, ctx context.Context, url, body, key string) *http.Response {
	t.Helper()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url+"/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	if key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	return resp
}

func readAll(t *testing.T, resp *http.Response) string {
	t.Helper()
	defer resp.Body.Close()
	b, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

func TestStreamPassthrough(t *testing.T) {
	e := newEnv(t, []http.Handler{fastFake("a")}, options{})
	resp := post(t, context.Background(), e.url, chatBody("sys", "hi", true), "")
	body := readAll(t, resp)

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	if ct := resp.Header.Get("Content-Type"); ct != "text/event-stream" {
		t.Errorf("Content-Type = %q", ct)
	}
	if got := resp.Header.Get("X-Router-Backend"); got != "b0" {
		t.Errorf("X-Router-Backend = %q, want b0", got)
	}
	// 5 tokens, the finish event and the usage event, then [DONE].
	if n := strings.Count(body, "data: {"); n != 7 {
		t.Errorf("got %d data events, want 7:\n%s", n, body)
	}
	if !strings.HasSuffix(body, "data: [DONE]\n\n") {
		t.Error("stream does not end with [DONE]")
	}
	if got := testutil.ToFloat64(e.metrics.Chunks.WithLabelValues("b0")); got != 7 {
		t.Errorf("chunks metric = %v, want 7", got)
	}
	if got := testutil.CollectAndCount(e.metrics.TTFT); got != 1 {
		t.Errorf("TTFT series = %d, want 1", got)
	}
	if got := testutil.ToFloat64(e.metrics.Tokens.WithLabelValues("b0", "completion")); got != 5 {
		t.Errorf("completion tokens = %v, want 5 from the usage event", got)
	}
}

func TestNonStreamUsage(t *testing.T) {
	e := newEnv(t, []http.Handler{fastFake("a")}, options{})
	resp := post(t, context.Background(), e.url, chatBody("sys", "hi", false), "")
	body := readAll(t, resp)
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status %d: %s", resp.StatusCode, body)
	}
	var v struct {
		Choices []struct {
			Message struct{ Content string }
		}
	}
	if err := json.Unmarshal([]byte(body), &v); err != nil || len(v.Choices) != 1 || v.Choices[0].Message.Content == "" {
		t.Errorf("unexpected body %s (%v)", body, err)
	}
	if got := testutil.ToFloat64(e.metrics.Tokens.WithLabelValues("b0", "completion")); got != 5 {
		t.Errorf("completion tokens = %v, want 5", got)
	}
}

// runDocs sends four questions about each of four documents to three
// backends and returns the cached prompt tokens the backends reported, and
// which backends served each document. Four documents on three backends keep
// round robin from lining documents up with backends by accident.
func runDocs(t *testing.T, policy string) (cached int64, servedBy map[int]map[string]bool) {
	fakes := []*fakellm.Server{fastFake("a"), fastFake("b"), fastFake("c")}
	e := newEnv(t, []http.Handler{fakes[0], fakes[1], fakes[2]}, options{policy: policy})
	servedBy = map[int]map[string]bool{}
	for q := range 4 {
		for d := range 4 {
			doc := strings.Repeat(fmt.Sprintf("Document %d says something long. ", d), 40)
			resp := post(t, context.Background(), e.url, chatBody(doc, fmt.Sprintf("question %d", q), false), "")
			readAll(t, resp)
			if servedBy[d] == nil {
				servedBy[d] = map[string]bool{}
			}
			servedBy[d][resp.Header.Get("X-Router-Backend")] = true
		}
	}
	for _, f := range fakes {
		cached += f.Stats().CachedTokens
	}
	return cached, servedBy
}

func TestPrefixRoutingReusesCache(t *testing.T) {
	prefixCached, servedBy := runDocs(t, "prefix")
	for d, backends := range servedBy {
		if len(backends) != 1 {
			t.Errorf("document %d was served by %d backends, want 1: %v", d, len(backends), backends)
		}
	}
	rrCached, _ := runDocs(t, "round_robin")
	if prefixCached <= rrCached {
		t.Errorf("prefix routing reused %d cached tokens, round robin %d; prefix should reuse more", prefixCached, rrCached)
	}
	t.Logf("cached prompt tokens: prefix %d, round robin %d", prefixCached, rrCached)
}

func TestRetryOnUnavailableBackend(t *testing.T) {
	loading := fastFake("loading")
	loading.SetReady(false)
	e := newEnv(t, []http.Handler{loading, fastFake("ok")}, options{policy: "round_robin", retries: 1, fall: 1})

	for i := range 2 {
		resp := post(t, context.Background(), e.url, chatBody("sys", "hi", false), "")
		if body := readAll(t, resp); resp.StatusCode != http.StatusOK {
			t.Fatalf("request %d: status %d: %s", i, resp.StatusCode, body)
		}
		if got := resp.Header.Get("X-Router-Backend"); got != "b1" {
			t.Errorf("request %d served by %s, want b1", i, got)
		}
	}
	if got := testutil.ToFloat64(e.metrics.Retries); got != 1 {
		t.Errorf("retries = %v, want 1 (the second request should skip the ejected backend)", got)
	}
	if e.backends[0].Healthy() {
		t.Error("backend answering 503 should be ejected with fall=1")
	}
}

func TestDeadBackends(t *testing.T) {
	e := newEnv(t, []http.Handler{nil, nil}, options{policy: "round_robin", retries: 1, fall: 5})
	resp := post(t, context.Background(), e.url, chatBody("s", "u", false), "")
	body := readAll(t, resp)
	if resp.StatusCode != http.StatusBadGateway {
		t.Errorf("status %d body %s, want 502 after both backends refused", resp.StatusCode, body)
	}
	if got := testutil.ToFloat64(e.metrics.Retries); got != 1 {
		t.Errorf("retries = %v, want 1", got)
	}
}

func TestRateLimit(t *testing.T) {
	e := newEnv(t, []http.Handler{fastFake("a")}, options{limiter: limit.New(1, 1, 0)})
	if resp := post(t, context.Background(), e.url, chatBody("s", "u", false), "k"); resp.StatusCode != http.StatusOK {
		t.Fatalf("first request: status %d", resp.StatusCode)
	}
	resp := post(t, context.Background(), e.url, chatBody("s", "u", false), "k")
	readAll(t, resp)
	if resp.StatusCode != http.StatusTooManyRequests || resp.Header.Get("Retry-After") != "1" {
		t.Errorf("status %d, Retry-After %q; want 429 and 1", resp.StatusCode, resp.Header.Get("Retry-After"))
	}
	// Another key has its own budget.
	if resp := post(t, context.Background(), e.url, chatBody("s", "u", false), "other"); resp.StatusCode != http.StatusOK {
		t.Errorf("other key: status %d", resp.StatusCode)
	}
}

func TestAPIKeysAndBackendAuth(t *testing.T) {
	var mu sync.Mutex
	var seen []string
	backend := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		seen = append(seen, r.Header.Get("Authorization"))
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		io.WriteString(w, `{"choices":[]}`)
	})
	e := newEnv(t, []http.Handler{backend}, options{apiKeys: map[string]bool{"secret": true}, backendKey: "backend-key"})

	if resp := post(t, context.Background(), e.url, chatBody("s", "u", false), ""); resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("no key: status %d, want 401", resp.StatusCode)
	}
	if resp := post(t, context.Background(), e.url, chatBody("s", "u", false), "wrong"); resp.StatusCode != http.StatusUnauthorized {
		t.Errorf("wrong key: status %d, want 401", resp.StatusCode)
	}
	if resp := post(t, context.Background(), e.url, chatBody("s", "u", false), "secret"); resp.StatusCode != http.StatusOK {
		t.Errorf("valid key: status %d, want 200", resp.StatusCode)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(seen) != 1 || seen[0] != "Bearer backend-key" {
		t.Errorf("backend saw Authorization %q, want only the backend's own key", seen)
	}
}

func TestClientDisconnectStopsGeneration(t *testing.T) {
	cfg := fakellm.DefaultConfig("slow")
	cfg.DecodeTPS = 20
	cfg.MaxTokens = 200 // ten seconds of tokens
	fake := fakellm.New(cfg)
	e := newEnv(t, []http.Handler{fake}, options{})

	ctx, cancel := context.WithCancel(context.Background())
	resp := post(t, ctx, e.url, chatBody("sys", "write a novel", true), "")
	if _, err := bufio.NewReader(resp.Body).ReadString('\n'); err != nil {
		t.Fatal(err)
	}
	cancel()
	resp.Body.Close()

	deadline := time.Now().Add(2 * time.Second)
	for fake.Stats().Cancelled == 0 {
		if time.Now().After(deadline) {
			t.Fatal("backend kept generating after the client left")
		}
		time.Sleep(10 * time.Millisecond)
	}
	for e.backends[0].Inflight() != 0 {
		if time.Now().After(deadline) {
			t.Fatal("slot was not released")
		}
		time.Sleep(10 * time.Millisecond)
	}
}

func TestQueueTimeoutWhenBackendsFull(t *testing.T) {
	cfg := fakellm.DefaultConfig("slow")
	cfg.DecodeTPS = 20
	cfg.MaxTokens = 200
	e := newEnv(t, []http.Handler{fakellm.New(cfg)}, options{maxInflight: 1, queueTimeout: 50 * time.Millisecond})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	first := post(t, ctx, e.url, chatBody("sys", "long", true), "")
	defer first.Body.Close()
	if _, err := bufio.NewReader(first.Body).ReadString('\n'); err != nil {
		t.Fatal(err)
	}

	resp := post(t, context.Background(), e.url, chatBody("sys", "short", false), "")
	body := readAll(t, resp)
	if resp.StatusCode != http.StatusServiceUnavailable || !strings.Contains(body, "queue_timeout") {
		t.Errorf("status %d body %s; want 503 queue_timeout", resp.StatusCode, body)
	}
	if got := testutil.ToFloat64(e.metrics.Rejected.WithLabelValues("queue_timeout")); got != 1 {
		t.Errorf("queue_timeout rejections = %v, want 1", got)
	}
}

func TestBadRequests(t *testing.T) {
	e := newEnv(t, []http.Handler{fastFake("a")}, options{})
	for _, body := range []string{`{`, `{"messages": []}`} {
		resp := post(t, context.Background(), e.url, body, "")
		readAll(t, resp)
		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("body %s: status %d, want 400", body, resp.StatusCode)
		}
	}
}

func TestModels(t *testing.T) {
	e := newEnv(t, []http.Handler{fastFake("a")}, options{})
	resp, err := http.Get(e.url + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	body := readAll(t, resp)
	if resp.StatusCode != http.StatusOK || !strings.Contains(body, `"fake"`) {
		t.Errorf("status %d body %s", resp.StatusCode, body)
	}
}
