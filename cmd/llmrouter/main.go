// Command llmrouter is a cache-aware load balancer for OpenAI-compatible LLM
// inference servers such as llama.cpp and vLLM.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net"
	"net/http"
	"net/http/pprof"
	"net/url"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/NikkiRe/llm-router/internal/balancer"
	"github.com/NikkiRe/llm-router/internal/config"
	"github.com/NikkiRe/llm-router/internal/health"
	"github.com/NikkiRe/llm-router/internal/limit"
	"github.com/NikkiRe/llm-router/internal/metrics"
	"github.com/NikkiRe/llm-router/internal/proxy"
)

// shutdownTimeout is how long open streams get to finish on SIGTERM.
const shutdownTimeout = 30 * time.Second

func main() {
	configPath := flag.String("config", "config.example.yaml", "path to the YAML configuration")
	policy := flag.String("policy", "", "override the routing policy: round_robin, least_loaded or prefix")
	listen := flag.String("listen", "", "override the API listen address")
	adminListen := flag.String("admin-listen", "", "override the admin listen address")
	logLevel := flag.String("log-level", "info", "debug, info, warn or error")
	flag.Parse()

	var level slog.Level
	if err := level.UnmarshalText([]byte(*logLevel)); err != nil {
		fmt.Fprintln(os.Stderr, "bad -log-level:", err)
		os.Exit(2)
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	cfg, err := config.Load(*configPath)
	if err == nil {
		if *policy != "" {
			cfg.Policy = *policy
		}
		if *listen != "" {
			cfg.Listen = *listen
		}
		if *adminListen != "" {
			cfg.AdminListen = *adminListen
		}
		err = cfg.Validate()
	}
	if err != nil {
		log.Error("invalid configuration", "err", err)
		os.Exit(2)
	}
	if err := run(cfg, log); err != nil {
		log.Error("router stopped", "err", err)
		os.Exit(1)
	}
}

func run(cfg config.Config, log *slog.Logger) error {
	backends := make([]*balancer.Backend, len(cfg.Backends))
	for i, b := range cfg.Backends {
		u, _ := url.Parse(b.URL) // checked by Validate
		backends[i] = balancer.NewBackend(b.Name, u, b.MaxInflight, cfg.Prefix.CacheChunks)
		backends[i].APIKey = b.APIKey
	}
	policy, err := balancer.NewPolicy(cfg.Policy, cfg.Prefix.MinMatchRatio, cfg.Prefix.LoadSlack)
	if err != nil {
		return err
	}
	pool := balancer.NewPool(backends, policy, balancer.Thresholds{Fall: cfg.Health.Fall, Rise: cfg.Health.Rise})

	reg := prometheus.NewRegistry()
	reg.MustRegister(collectors.NewGoCollector(), collectors.NewProcessCollector(collectors.ProcessCollectorOpts{}))
	m := metrics.New(reg, pool)

	transport := &http.Transport{
		DialContext:           (&net.Dialer{Timeout: 5 * time.Second, KeepAlive: 30 * time.Second}).DialContext,
		MaxIdleConns:          256,
		MaxIdleConnsPerHost:   64,
		IdleConnTimeout:       90 * time.Second,
		ResponseHeaderTimeout: cfg.ResponseHeaderTimeout,
	}
	limiter := limit.New(cfg.Limits.RequestsPerSecond, cfg.Limits.Burst, cfg.Limits.MaxConcurrent)
	keys := make(map[string]bool, len(cfg.Limits.APIKeys))
	for _, k := range cfg.Limits.APIKeys {
		keys[k] = true
	}
	api := proxy.New(proxy.Config{
		ChunkSize:    cfg.Prefix.ChunkSize,
		QueueTimeout: cfg.QueueTimeout,
		Retries:      cfg.Retries,
		MaxBodyBytes: cfg.MaxBodyBytes,
		APIKeys:      keys,
	}, pool, limiter, m, transport, log)

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	checker := &health.Checker{
		Pool:     pool,
		Client:   &http.Client{Transport: transport},
		Path:     cfg.Health.Path,
		Interval: cfg.Health.Interval,
		Timeout:  cfg.Health.Timeout,
		Log:      log,
	}
	go checker.Run(ctx)
	go sweepLimiter(ctx, limiter)

	srv := &http.Server{Addr: cfg.Listen, Handler: api, ReadHeaderTimeout: 10 * time.Second}
	admin := &http.Server{Addr: cfg.AdminListen, Handler: adminMux(reg, pool), ReadHeaderTimeout: 10 * time.Second}
	errc := make(chan error, 2)
	go func() { errc <- srv.ListenAndServe() }()
	go func() { errc <- admin.ListenAndServe() }()
	log.Info("router started", "listen", cfg.Listen, "admin", cfg.AdminListen,
		"policy", policy.Name(), "backends", len(backends))

	select {
	case err := <-errc:
		return err
	case <-ctx.Done():
	}
	log.Info("shutting down, waiting for open requests", "timeout", shutdownTimeout)
	shutdownCtx, cancel := context.WithTimeout(context.Background(), shutdownTimeout)
	defer cancel()
	return errors.Join(srv.Shutdown(shutdownCtx), admin.Shutdown(shutdownCtx))
}

func adminMux(reg *prometheus.Registry, pool *balancer.Pool) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("GET /metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("ok\n"))
	})
	mux.HandleFunc("GET /backends", func(w http.ResponseWriter, _ *http.Request) {
		type status struct {
			Name        string `json:"name"`
			URL         string `json:"url"`
			Healthy     bool   `json:"healthy"`
			Inflight    int    `json:"inflight"`
			MaxInflight int    `json:"max_inflight"`
		}
		var out []status
		for _, b := range pool.Backends() {
			out = append(out, status{b.Name, b.URL.String(), b.Healthy(), b.Inflight(), b.MaxInflight})
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{"policy": pool.Policy().Name(), "backends": out})
	})
	mux.HandleFunc("/debug/pprof/", pprof.Index)
	mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
	mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
	mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
	mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
	return mux
}

func sweepLimiter(ctx context.Context, l *limit.Limiter) {
	t := time.NewTicker(time.Minute)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			l.Sweep(10 * time.Minute)
		}
	}
}
