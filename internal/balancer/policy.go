package balancer

import (
	"fmt"
	"sync/atomic"
)

// NewPolicy returns the policy with the given name.
func NewPolicy(name string, minMatchRatio float64, loadSlack int) (Policy, error) {
	switch name {
	case "round_robin":
		return &RoundRobin{}, nil
	case "least_loaded":
		return &LeastLoaded{}, nil
	case "prefix":
		return &PrefixAware{MinMatchRatio: minMatchRatio, LoadSlack: loadSlack}, nil
	default:
		return nil, fmt.Errorf("unknown policy %q (want round_robin, least_loaded or prefix)", name)
	}
}

// RoundRobin sends requests to backends in turn, ignoring load and cache.
type RoundRobin struct {
	next atomic.Uint64
}

func (*RoundRobin) Name() string { return "round_robin" }

func (p *RoundRobin) Pick(_ Request, c []*Backend) Decision {
	b := c[int(p.next.Add(1)-1)%len(c)]
	return Decision{Backend: b, Reason: "round_robin"}
}

// LeastLoaded sends a request to the backend with the fewest requests in
// flight. Ties rotate, so an idle pool still spreads traffic.
type LeastLoaded struct {
	next atomic.Uint64
}

func (*LeastLoaded) Name() string { return "least_loaded" }

func (p *LeastLoaded) Pick(_ Request, c []*Backend) Decision {
	start := int(p.next.Add(1)-1) % len(c)
	best := c[start]
	for i := 1; i < len(c); i++ {
		b := c[(start+i)%len(c)]
		if b.Inflight() < best.Inflight() {
			best = b
		}
	}
	return Decision{Backend: best, Reason: "least_loaded"}
}

// PrefixAware sends a request to the backend that most likely holds its
// prompt prefix in the KV cache, so the backend skips prefill for that part.
// Affinity must not overload one backend, so a backend is eligible only if it
// runs at most LoadSlack more requests than the least loaded one. When no
// eligible backend has cached at least MinMatchRatio of the prompt, the
// request goes to the least loaded backend.
type PrefixAware struct {
	MinMatchRatio float64
	LoadSlack     int

	fallback LeastLoaded
}

func (*PrefixAware) Name() string { return "prefix" }

func (p *PrefixAware) Pick(req Request, c []*Backend) Decision {
	minLoad := c[0].Inflight()
	for _, b := range c[1:] {
		minLoad = min(minLoad, b.Inflight())
	}

	var best *Backend
	bestMatch := 0
	for _, b := range c {
		if b.Inflight() > minLoad+p.LoadSlack {
			continue
		}
		if m := b.index.Match(req.Hashes); m > bestMatch {
			best, bestMatch = b, m
		}
	}
	if best != nil && float64(bestMatch) >= p.MinMatchRatio*float64(len(req.Hashes)) {
		return Decision{Backend: best, Reason: "prefix", Matched: bestMatch}
	}

	d := p.fallback.Pick(req, c)
	d.Matched = d.Backend.index.Match(req.Hashes)
	return d
}
