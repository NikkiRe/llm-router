// Command fakellm runs a simulated OpenAI-compatible inference server with a
// KV prefix cache, for benchmarks and demos without a GPU.
package main

import (
	"context"
	"flag"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/NikkiRe/llm-router/internal/fakellm"
)

func main() {
	d := fakellm.DefaultConfig("fake")
	addr := flag.String("addr", ":8081", "listen address")
	name := flag.String("name", d.Name, "server name, shown in /v1/models")
	slots := flag.Int("slots", d.Slots, "requests served at once")
	prefill := flag.Float64("prefill-tps", d.PrefillTPS, "uncached prompt tokens per second")
	decode := flag.Float64("decode-tps", d.DecodeTPS, "generated tokens per second, per request")
	cache := flag.Int("cache-tokens", d.CacheTokens, "KV cache capacity in tokens")
	maxTokens := flag.Int("max-tokens", d.MaxTokens, "tokens generated when a request sets no limit")
	loadDelay := flag.Duration("load-delay", 0, "answer 503 on /health for this long, like a loading model")
	flag.Parse()

	cfg := d
	cfg.Name, cfg.Slots, cfg.PrefillTPS, cfg.DecodeTPS, cfg.CacheTokens, cfg.MaxTokens =
		*name, *slots, *prefill, *decode, *cache, *maxTokens
	srv := fakellm.New(cfg)
	if *loadDelay > 0 {
		srv.SetReady(false)
		time.AfterFunc(*loadDelay, func() { srv.SetReady(true) })
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	hs := &http.Server{Addr: *addr, Handler: srv, ReadHeaderTimeout: 10 * time.Second}
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = hs.Shutdown(shutdownCtx)
	}()
	slog.Info("fakellm listening", "addr", *addr, "name", cfg.Name, "slots", cfg.Slots,
		"prefill_tps", cfg.PrefillTPS, "decode_tps", cfg.DecodeTPS)
	if err := hs.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		slog.Error("server failed", "err", err)
		os.Exit(1)
	}
}
