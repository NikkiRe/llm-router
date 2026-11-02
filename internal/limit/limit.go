// Package limit applies per-API-key request rates and concurrency caps.
package limit

import (
	"errors"
	"math"
	"sync"
	"time"
)

var (
	// ErrRateLimited means the key spent its request budget.
	ErrRateLimited = errors.New("rate limit exceeded")
	// ErrTooManyConcurrent means the key has too many requests in flight.
	ErrTooManyConcurrent = errors.New("too many concurrent requests")
)

// Bucket is a token bucket: it holds up to burst tokens and refills at rate
// tokens per second. Each request takes one token.
type Bucket struct {
	rate, burst float64
	tokens      float64
	last        time.Time
}

// NewBucket returns a full bucket.
func NewBucket(rate float64, burst int, now time.Time) *Bucket {
	return &Bucket{rate: rate, burst: float64(burst), tokens: float64(burst), last: now}
}

// Take removes one token if there is one. Otherwise it reports how long until
// the next token arrives.
func (b *Bucket) Take(now time.Time) (ok bool, retryAfter time.Duration) {
	if elapsed := now.Sub(b.last).Seconds(); elapsed > 0 {
		b.tokens = math.Min(b.burst, b.tokens+elapsed*b.rate)
		b.last = now
	}
	if b.tokens >= 1 {
		b.tokens--
		return true, 0
	}
	wait := (1 - b.tokens) / b.rate
	return false, time.Duration(math.Ceil(wait * float64(time.Second)))
}

type keyState struct {
	bucket   *Bucket
	active   int
	lastSeen time.Time
}

// Limiter tracks every API key. A zero Rate or MaxConcurrent disables that
// limit.
type Limiter struct {
	Rate          float64
	Burst         int
	MaxConcurrent int
	// Now returns the current time; tests replace it.
	Now func() time.Time

	mu   sync.Mutex
	keys map[string]*keyState
}

// New returns a limiter with the given limits.
func New(rate float64, burst, maxConcurrent int) *Limiter {
	if burst < 1 {
		burst = max(1, int(math.Ceil(rate)))
	}
	return &Limiter{Rate: rate, Burst: burst, MaxConcurrent: maxConcurrent, Now: time.Now}
}

// Acquire admits one request for key. On success release must be called when
// the request ends. On ErrRateLimited retryAfter says when to try again.
func (l *Limiter) Acquire(key string) (release func(), retryAfter time.Duration, err error) {
	l.mu.Lock()
	defer l.mu.Unlock()
	now := l.Now()
	if l.keys == nil {
		l.keys = make(map[string]*keyState)
	}
	s, ok := l.keys[key]
	if !ok {
		s = &keyState{}
		if l.Rate > 0 {
			s.bucket = NewBucket(l.Rate, l.Burst, now)
		}
		l.keys[key] = s
	}
	s.lastSeen = now

	// Check concurrency first so a rejected request does not burn a token.
	if l.MaxConcurrent > 0 && s.active >= l.MaxConcurrent {
		return nil, 0, ErrTooManyConcurrent
	}
	if s.bucket != nil {
		if ok, wait := s.bucket.Take(now); !ok {
			return nil, wait, ErrRateLimited
		}
	}
	s.active++

	var once sync.Once
	return func() {
		once.Do(func() {
			l.mu.Lock()
			s.active--
			l.mu.Unlock()
		})
	}, 0, nil
}

// Sweep forgets keys that have no requests in flight and were last seen
// longer than idle ago, so the map does not grow with every key ever seen.
// A forgotten key starts again with a full bucket.
func (l *Limiter) Sweep(idle time.Duration) {
	l.mu.Lock()
	defer l.mu.Unlock()
	cutoff := l.Now().Add(-idle)
	for k, s := range l.keys {
		if s.active == 0 && s.lastSeen.Before(cutoff) {
			delete(l.keys, k)
		}
	}
}

// Keys returns how many keys are tracked.
func (l *Limiter) Keys() int {
	l.mu.Lock()
	defer l.mu.Unlock()
	return len(l.keys)
}
