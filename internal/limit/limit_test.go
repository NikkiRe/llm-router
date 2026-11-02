package limit

import (
	"errors"
	"testing"
	"time"
)

type clock struct{ t time.Time }

func (c *clock) now() time.Time          { return c.t }
func (c *clock) advance(d time.Duration) { c.t = c.t.Add(d) }

func newLimiter(rate float64, burst, maxConcurrent int) (*Limiter, *clock) {
	c := &clock{t: time.Unix(1_700_000_000, 0)}
	l := New(rate, burst, maxConcurrent)
	l.Now = c.now
	return l, c
}

func TestBucketBurstThenRefill(t *testing.T) {
	l, c := newLimiter(2, 3, 0)
	for i := range 3 {
		if _, _, err := l.Acquire("k"); err != nil {
			t.Fatalf("request %d within burst: %v", i, err)
		}
	}
	_, wait, err := l.Acquire("k")
	if !errors.Is(err, ErrRateLimited) {
		t.Fatalf("err = %v, want ErrRateLimited", err)
	}
	if wait != 500*time.Millisecond {
		t.Errorf("retryAfter = %v, want 500ms at 2 requests per second", wait)
	}

	c.advance(500 * time.Millisecond)
	if _, _, err := l.Acquire("k"); err != nil {
		t.Errorf("after one refill interval: %v", err)
	}
}

func TestBucketDoesNotOverfill(t *testing.T) {
	l, c := newLimiter(1, 2, 0)
	c.advance(time.Hour)
	for range 2 {
		if _, _, err := l.Acquire("k"); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := l.Acquire("k"); !errors.Is(err, ErrRateLimited) {
		t.Errorf("a long idle period must not add more than burst tokens, err = %v", err)
	}
}

func TestKeysAreIndependent(t *testing.T) {
	l, _ := newLimiter(1, 1, 0)
	if _, _, err := l.Acquire("a"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := l.Acquire("b"); err != nil {
		t.Errorf("key b was limited by key a: %v", err)
	}
}

func TestConcurrencyCap(t *testing.T) {
	l, _ := newLimiter(0, 0, 2)
	r1, _, _ := l.Acquire("k")
	if _, _, err := l.Acquire("k"); err != nil {
		t.Fatal(err)
	}
	if _, _, err := l.Acquire("k"); !errors.Is(err, ErrTooManyConcurrent) {
		t.Fatalf("err = %v, want ErrTooManyConcurrent", err)
	}
	r1()
	r1() // a second release must not free another slot
	if _, _, err := l.Acquire("k"); err != nil {
		t.Errorf("after release: %v", err)
	}
	if _, _, err := l.Acquire("k"); !errors.Is(err, ErrTooManyConcurrent) {
		t.Errorf("double release freed an extra slot, err = %v", err)
	}
}

func TestConcurrencyRejectionKeepsToken(t *testing.T) {
	l, _ := newLimiter(1, 2, 1) // two tokens, one request at a time
	release, _, err := l.Acquire("k")
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := l.Acquire("k"); !errors.Is(err, ErrTooManyConcurrent) {
		t.Fatalf("err = %v, want ErrTooManyConcurrent", err)
	}
	release()
	// The clock has not moved, so this succeeds only if the rejected
	// request above left the second token in the bucket.
	if _, _, err := l.Acquire("k"); err != nil {
		t.Errorf("rejected request burned a token: %v", err)
	}
}

func TestSweepDropsIdleKeys(t *testing.T) {
	l, c := newLimiter(1, 1, 0)
	release, _, _ := l.Acquire("busy")
	releaseIdle, _, _ := l.Acquire("idle")
	releaseIdle()
	c.advance(time.Hour)

	l.Sweep(10 * time.Minute)
	if l.Keys() != 1 {
		t.Fatalf("Keys = %d, want 1 (the key with a request in flight stays)", l.Keys())
	}
	release()
	l.Sweep(10 * time.Minute)
	if l.Keys() != 0 {
		t.Errorf("Keys = %d, want 0", l.Keys())
	}
}
