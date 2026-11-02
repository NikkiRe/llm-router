// Package fakellm is a simulated OpenAI-compatible inference server with a
// KV prefix cache. It stands in for llama.cpp in tests, CI and benchmarks
// that have no GPU.
//
// The model of cost is deliberately simple: prefill of uncached prompt tokens
// runs one request at a time at PrefillTPS (prefill is compute-bound), and
// decoding runs in parallel across Slots at DecodeTPS per request.
package fakellm

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/NikkiRe/llm-router/internal/openai"
	"github.com/NikkiRe/llm-router/internal/prefix"
)

// Config describes the simulated server.
type Config struct {
	Name          string
	Slots         int     // requests served at once, like llama-server -np
	PrefillTPS    float64 // uncached prompt tokens per second
	DecodeTPS     float64 // generated tokens per second, per request
	CacheTokens   int     // KV cache capacity
	BlockTokens   int     // KV cache block size
	BytesPerToken int
	MaxTokens     int // used when a request sets no limit
}

// DefaultConfig is roughly a small model on one consumer GPU.
func DefaultConfig(name string) Config {
	return Config{
		Name:          name,
		Slots:         4,
		PrefillTPS:    4000,
		DecodeTPS:     40,
		CacheTokens:   32768,
		BlockTokens:   16,
		BytesPerToken: 4,
		MaxTokens:     16,
	}
}

// Stats counts what the server did.
type Stats struct {
	Requests     int64 `json:"requests"`
	PromptTokens int64 `json:"prompt_tokens"`
	CachedTokens int64 `json:"cached_tokens"`
	Cancelled    int64 `json:"cancelled"`
}

// Server is the simulated backend.
type Server struct {
	cfg   Config
	slots chan struct{}
	// gpu serialises prefill. It is a channel, not a mutex, so a request
	// waiting for it can give up when its client leaves.
	gpu   chan struct{}
	cache *prefix.Index
	ready atomic.Bool
	mux   *http.ServeMux
	seq   atomic.Int64

	requests, promptTokens, cachedTokens, cancelled atomic.Int64
}

// New returns a ready server.
func New(cfg Config) *Server {
	s := &Server{
		cfg:   cfg,
		slots: make(chan struct{}, max(cfg.Slots, 1)),
		gpu:   make(chan struct{}, 1),
		cache: prefix.NewIndex(max(cfg.CacheTokens/cfg.BlockTokens, 1)),
		mux:   http.NewServeMux(),
	}
	s.ready.Store(true)
	s.mux.HandleFunc("GET /health", s.health)
	s.mux.HandleFunc("GET /v1/models", s.models)
	s.mux.HandleFunc("GET /stats", s.stats)
	s.mux.HandleFunc("POST /v1/chat/completions", s.complete)
	s.mux.HandleFunc("POST /v1/completions", s.complete)
	return s
}

func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) { s.mux.ServeHTTP(w, r) }

// SetReady switches the server between serving and "loading model", when it
// answers 503 like llama.cpp does.
func (s *Server) SetReady(ready bool) { s.ready.Store(ready) }

// Stats returns the counters so far.
func (s *Server) Stats() Stats {
	return Stats{
		Requests:     s.requests.Load(),
		PromptTokens: s.promptTokens.Load(),
		CachedTokens: s.cachedTokens.Load(),
		Cancelled:    s.cancelled.Load(),
	}
}

func (s *Server) health(w http.ResponseWriter, _ *http.Request) {
	if !s.ready.Load() {
		writeJSON(w, http.StatusServiceUnavailable, map[string]any{"status": "loading model"})
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{"status": "ok"})
}

func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   []any{map[string]any{"id": "fake", "object": "model", "owned_by": s.cfg.Name}},
	})
}

func (s *Server) stats(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, http.StatusOK, s.Stats())
}

func (s *Server) complete(w http.ResponseWriter, r *http.Request) {
	if !s.ready.Load() {
		openai.WriteError(w, http.StatusServiceUnavailable, "unavailable", "loading model")
		return
	}
	body, err := io.ReadAll(r.Body)
	if err != nil {
		openai.WriteError(w, http.StatusBadRequest, "bad_request", "cannot read body")
		return
	}
	req, err := openai.Parse(r.URL.Path, body)
	if err != nil {
		openai.WriteError(w, http.StatusBadRequest, "bad_request", err.Error())
		return
	}
	ctx := r.Context()

	select {
	case s.slots <- struct{}{}:
	case <-ctx.Done():
		s.cancelled.Add(1)
		return
	}
	defer func() { <-s.slots }()

	// Prefill: only the part of the prompt missing from the cache costs time.
	promptTokens := (len(req.Prompt) + s.cfg.BytesPerToken - 1) / s.cfg.BytesPerToken
	hashes := prefix.Hashes(req.Prompt, s.cfg.BlockTokens*s.cfg.BytesPerToken)
	select {
	case s.gpu <- struct{}{}:
	case <-ctx.Done():
		s.cancelled.Add(1)
		return
	}
	cached := min(s.cache.Match(hashes)*s.cfg.BlockTokens, promptTokens)
	prefill := tokensTime(promptTokens-cached, s.cfg.PrefillTPS)
	if !sleep(r, prefill) {
		<-s.gpu
		s.cancelled.Add(1)
		return
	}
	s.cache.Insert(hashes)
	<-s.gpu

	s.requests.Add(1)
	s.promptTokens.Add(int64(promptTokens))
	s.cachedTokens.Add(int64(cached))

	n := req.MaxTokens
	if n <= 0 {
		n = s.cfg.MaxTokens
	}
	g := generation{
		id:      fmt.Sprintf("cmpl-%s-%d", s.cfg.Name, s.seq.Add(1)),
		chat:    strings.HasSuffix(r.URL.Path, "/chat/completions"),
		tokens:  n,
		perTok:  tokensTime(1, s.cfg.DecodeTPS),
		created: time.Now().Unix(),
		usage:   usage(promptTokens, cached, n),
	}
	if req.Stream {
		if !g.stream(w, r, req.IncludeUsage) {
			s.cancelled.Add(1)
		}
		return
	}
	if !sleep(r, time.Duration(n)*g.perTok) {
		s.cancelled.Add(1)
		return
	}
	writeJSON(w, http.StatusOK, g.full())
}

type generation struct {
	id      string
	chat    bool
	tokens  int
	perTok  time.Duration
	created int64
	usage   openai.Usage
}

var words = []string{" the", " router", " keeps", " prompts", " near", " their", " cache", "."}

func (g generation) text() string {
	var sb strings.Builder
	for i := range g.tokens {
		sb.WriteString(words[i%len(words)])
	}
	return sb.String()
}

func (g generation) full() map[string]any {
	choice := map[string]any{"index": 0, "finish_reason": "length"}
	object := "text_completion"
	if g.chat {
		object = "chat.completion"
		choice["message"] = map[string]any{"role": "assistant", "content": g.text()}
	} else {
		choice["text"] = g.text()
	}
	return map[string]any{
		"id": g.id, "object": object, "created": g.created, "model": "fake",
		"choices": []any{choice}, "usage": g.usage,
	}
}

func (g generation) chunk(token string, finish any) map[string]any {
	choice := map[string]any{"index": 0, "finish_reason": finish}
	object := "text_completion"
	if g.chat {
		object = "chat.completion.chunk"
		delta := map[string]any{}
		if token != "" {
			delta["content"] = token
		}
		choice["delta"] = delta
	} else {
		choice["text"] = token
	}
	return map[string]any{
		"id": g.id, "object": object, "created": g.created, "model": "fake",
		"choices": []any{choice},
	}
}

// stream writes one event per token. It returns false if the client left.
func (g generation) stream(w http.ResponseWriter, r *http.Request, includeUsage bool) bool {
	rc := http.NewResponseController(w)
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.WriteHeader(http.StatusOK)

	send := func(v any) bool {
		data, _ := json.Marshal(v)
		if _, err := fmt.Fprintf(w, "data: %s\n\n", data); err != nil {
			return false
		}
		return rc.Flush() == nil
	}
	for i := range g.tokens {
		if i > 0 && !sleep(r, g.perTok) {
			return false
		}
		if !send(g.chunk(words[i%len(words)], nil)) {
			return false
		}
	}
	if !send(g.chunk("", "length")) {
		return false
	}
	if includeUsage {
		u := g.chunk("", nil)
		u["choices"] = []any{}
		u["usage"] = g.usage
		if !send(u) {
			return false
		}
	}
	_, err := io.WriteString(w, "data: [DONE]\n\n")
	return err == nil && rc.Flush() == nil
}

func usage(prompt, cached, completion int) openai.Usage {
	return openai.Usage{
		PromptTokens:        prompt,
		CompletionTokens:    completion,
		TotalTokens:         prompt + completion,
		PromptTokensDetails: &openai.PromptTokensDetails{CachedTokens: cached},
	}
}

func tokensTime(tokens int, perSecond float64) time.Duration {
	if tokens <= 0 || perSecond <= 0 {
		return 0
	}
	return time.Duration(float64(tokens) / perSecond * float64(time.Second))
}

// sleep waits for d unless the client leaves first, in which case it
// returns false.
func sleep(r *http.Request, d time.Duration) bool {
	if d <= 0 {
		return r.Context().Err() == nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-t.C:
		return true
	case <-r.Context().Done():
		return false
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
