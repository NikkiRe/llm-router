// Package balancer holds the pool of LLM backends, decides which one serves
// each request and limits how many requests each backend runs at once.
package balancer

import (
	"context"
	"errors"
	"net/url"
	"sync"
	"sync/atomic"

	"github.com/NikkiRe/llm-router/internal/prefix"
)

// ErrNoBackend means no healthy backend is left to try.
var ErrNoBackend = errors.New("no healthy backend")

// Backend is one inference server, such as a llama.cpp or vLLM instance.
type Backend struct {
	Name        string
	URL         *url.URL
	APIKey      string
	MaxInflight int

	index *prefix.Index

	// inflight and healthy change only under Pool.mu; they are atomic so that
	// metrics and status pages can read them without the lock.
	inflight atomic.Int64
	healthy  atomic.Bool

	// Consecutive check results, guarded by Pool.mu.
	fails, oks int
}

// NewBackend returns a healthy backend whose prefix index holds up to
// cacheChunks chunk hashes.
func NewBackend(name string, u *url.URL, maxInflight, cacheChunks int) *Backend {
	b := &Backend{Name: name, URL: u, MaxInflight: maxInflight, index: prefix.NewIndex(cacheChunks)}
	b.healthy.Store(true)
	return b
}

// Inflight returns the number of requests the backend is running.
func (b *Backend) Inflight() int { return int(b.inflight.Load()) }

// Healthy reports whether the backend receives traffic.
func (b *Backend) Healthy() bool { return b.healthy.Load() }

// Request describes what routing needs to know about a request.
type Request struct {
	// Hashes are the prompt's prefix chunk hashes, see prefix.Hashes.
	Hashes []uint64
	// Exclude lists backends that already failed this request.
	Exclude []*Backend
}

// Decision is the policy's choice and why it was made.
type Decision struct {
	Backend *Backend
	// Reason is "prefix", "least_loaded" or "round_robin".
	Reason string
	// Matched is how many of the prompt's chunks the chosen backend was
	// believed to have cached.
	Matched int
}

// Policy chooses a backend. Pick is called with the pool locked and with
// candidates that are healthy and below their in-flight limit.
type Policy interface {
	Name() string
	Pick(req Request, candidates []*Backend) Decision
}

// Thresholds sets how many consecutive failed or passed checks flip a
// backend's health.
type Thresholds struct {
	Fall int
	Rise int
}

// Pool owns the backends and their admission.
type Pool struct {
	policy     Policy
	thresholds Thresholds

	mu       sync.Mutex
	backends []*Backend
	// wake is closed and replaced whenever a slot frees up or a backend
	// comes back, so that waiting requests try again.
	wake chan struct{}
}

// NewPool returns a pool that routes with policy.
func NewPool(backends []*Backend, policy Policy, t Thresholds) *Pool {
	return &Pool{
		policy:     policy,
		thresholds: Thresholds{Fall: max(t.Fall, 1), Rise: max(t.Rise, 1)},
		backends:   backends,
		wake:       make(chan struct{}),
	}
}

// Backends returns the backends in configuration order.
func (p *Pool) Backends() []*Backend { return p.backends }

// Policy returns the routing policy.
func (p *Pool) Policy() Policy { return p.policy }

// Lease is a reserved slot on a backend. Release must be called once the
// request is finished.
type Lease struct {
	Decision
	pool *Pool
	once sync.Once
}

// Release frees the slot. It is safe to call more than once.
func (l *Lease) Release() {
	l.once.Do(func() {
		l.pool.mu.Lock()
		l.Backend.inflight.Add(-1)
		l.pool.signalLocked()
		l.pool.mu.Unlock()
	})
}

// Acquire picks a backend for req and reserves a slot on it. When every
// healthy backend is full it waits for a free slot until ctx ends, which is
// how the router applies backpressure instead of overloading the GPUs.
func (p *Pool) Acquire(ctx context.Context, req Request) (*Lease, error) {
	for {
		p.mu.Lock()
		candidates, anyHealthy := p.candidatesLocked(req.Exclude)
		if len(candidates) > 0 {
			d := p.policy.Pick(req, candidates)
			d.Backend.inflight.Add(1)
			d.Backend.index.Insert(req.Hashes)
			p.mu.Unlock()
			return &Lease{Decision: d, pool: p}, nil
		}
		wake := p.wake
		p.mu.Unlock()
		if !anyHealthy {
			return nil, ErrNoBackend
		}
		select {
		case <-wake:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

func (p *Pool) candidatesLocked(exclude []*Backend) (candidates []*Backend, anyHealthy bool) {
	for _, b := range p.backends {
		if !b.Healthy() || excluded(b, exclude) {
			continue
		}
		anyHealthy = true
		if b.MaxInflight <= 0 || b.Inflight() < b.MaxInflight {
			candidates = append(candidates, b)
		}
	}
	return candidates, anyHealthy
}

func excluded(b *Backend, list []*Backend) bool {
	for _, x := range list {
		if x == b {
			return true
		}
	}
	return false
}

// Observe records the result of a health check or of a proxied request. It
// returns true if the backend's health changed.
func (p *Pool) Observe(b *Backend, ok bool) (changed bool) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if ok {
		b.fails = 0
		b.oks++
		if !b.Healthy() && b.oks >= p.thresholds.Rise {
			// The backend was down, so it has probably restarted with an
			// empty KV cache: whatever we believed it had cached is gone.
			b.index.Reset()
			b.healthy.Store(true)
			p.signalLocked()
			return true
		}
		return false
	}
	b.oks = 0
	b.fails++
	if b.Healthy() && b.fails >= p.thresholds.Fall {
		b.healthy.Store(false)
		// Waiters may now have no healthy backend left and must find out.
		p.signalLocked()
		return true
	}
	return false
}

func (p *Pool) signalLocked() {
	close(p.wake)
	p.wake = make(chan struct{})
}
