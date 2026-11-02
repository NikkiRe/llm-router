// Package health probes backends and ejects the ones that stop answering.
package health

import (
	"context"
	"log/slog"
	"net/http"
	"strings"
	"sync"
	"time"

	"github.com/NikkiRe/llm-router/internal/balancer"
)

// Checker polls each backend's health endpoint. llama.cpp's /health answers
// 503 while the model loads and 200 once it serves, which is exactly the
// signal the router needs.
type Checker struct {
	Pool     *balancer.Pool
	Client   *http.Client
	Path     string
	Interval time.Duration
	Timeout  time.Duration
	Log      *slog.Logger
}

// Run checks all backends right away and then every Interval until ctx ends.
func (c *Checker) Run(ctx context.Context) {
	t := time.NewTicker(c.Interval)
	defer t.Stop()
	for {
		c.CheckAll(ctx)
		select {
		case <-ctx.Done():
			return
		case <-t.C:
		}
	}
}

// CheckAll probes every backend in parallel and waits for the results.
func (c *Checker) CheckAll(ctx context.Context) {
	var wg sync.WaitGroup
	for _, b := range c.Pool.Backends() {
		wg.Go(func() {
			ok := c.probe(ctx, b)
			if ctx.Err() != nil {
				return // shutting down; the failure is ours
			}
			if c.Pool.Observe(b, ok) {
				if ok {
					c.Log.Info("backend is back", "backend", b.Name)
				} else {
					c.Log.Warn("backend ejected", "backend", b.Name)
				}
			}
		})
	}
	wg.Wait()
}

func (c *Checker) probe(ctx context.Context, b *balancer.Backend) bool {
	ctx, cancel := context.WithTimeout(ctx, c.Timeout)
	defer cancel()
	u := *b.URL
	u.Path = strings.TrimRight(b.URL.Path, "/") + c.Path
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return false
	}
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}
	resp, err := c.Client.Do(req)
	if err != nil {
		return false
	}
	resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}
