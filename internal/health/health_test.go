package health

import (
	"context"
	"log/slog"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"

	"github.com/NikkiRe/llm-router/internal/balancer"
	"github.com/NikkiRe/llm-router/internal/fakellm"
)

func TestCheckerEjectsAndReadmits(t *testing.T) {
	fake := fakellm.New(fakellm.DefaultConfig("a"))
	srv := httptest.NewServer(fake)
	defer srv.Close()
	u, _ := url.Parse(srv.URL)
	b := balancer.NewBackend("a", u, 0, 16)
	pool := balancer.NewPool([]*balancer.Backend{b}, &balancer.LeastLoaded{}, balancer.Thresholds{Fall: 2, Rise: 1})
	c := &Checker{
		Pool:     pool,
		Client:   http.DefaultClient,
		Path:     "/health",
		Interval: time.Hour,
		Timeout:  time.Second,
		Log:      slog.New(slog.DiscardHandler),
	}
	ctx := context.Background()

	fake.SetReady(false) // model still loading
	c.CheckAll(ctx)
	if !b.Healthy() {
		t.Fatal("one failed check must not eject with fall=2")
	}
	c.CheckAll(ctx)
	if b.Healthy() {
		t.Fatal("backend should be ejected after two failed checks")
	}

	fake.SetReady(true)
	c.CheckAll(ctx)
	if !b.Healthy() {
		t.Fatal("backend should be readmitted after one passed check")
	}
}

func TestCheckerTimesOut(t *testing.T) {
	block := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { <-block }))
	defer srv.Close()
	defer close(block)
	u, _ := url.Parse(srv.URL)
	b := balancer.NewBackend("hung", u, 0, 16)
	pool := balancer.NewPool([]*balancer.Backend{b}, &balancer.LeastLoaded{}, balancer.Thresholds{Fall: 1, Rise: 1})
	c := &Checker{Pool: pool, Client: http.DefaultClient, Path: "/health", Interval: time.Hour,
		Timeout: 30 * time.Millisecond, Log: slog.New(slog.DiscardHandler)}

	start := time.Now()
	c.CheckAll(context.Background())
	if b.Healthy() {
		t.Error("a backend that never answers should be ejected")
	}
	if time.Since(start) > time.Second {
		t.Error("check did not respect its timeout")
	}
}
