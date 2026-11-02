package balancer

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"strings"
	"testing"
	"time"

	"github.com/NikkiRe/llm-router/internal/prefix"
)

func newBackends(n, maxInflight int) []*Backend {
	out := make([]*Backend, n)
	for i := range out {
		u, _ := url.Parse(fmt.Sprintf("http://backend-%d", i))
		out[i] = NewBackend(fmt.Sprintf("b%d", i), u, maxInflight, 1000)
	}
	return out
}

func hashes(text string) []uint64 { return prefix.Hashes(text, 8) }

func mustAcquire(t *testing.T, p *Pool, req Request) *Lease {
	t.Helper()
	l, err := p.Acquire(context.Background(), req)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

func TestRoundRobinCycles(t *testing.T) {
	p := NewPool(newBackends(3, 0), &RoundRobin{}, Thresholds{})
	var got []string
	for range 6 {
		got = append(got, mustAcquire(t, p, Request{}).Backend.Name)
	}
	if want := "b0 b1 b2 b0 b1 b2"; strings.Join(got, " ") != want {
		t.Errorf("got %v, want %s", got, want)
	}
}

func TestLeastLoadedPicksIdleBackend(t *testing.T) {
	p := NewPool(newBackends(3, 0), &LeastLoaded{}, Thresholds{})
	a := mustAcquire(t, p, Request{})
	b := mustAcquire(t, p, Request{})
	c := mustAcquire(t, p, Request{})
	if a.Backend == b.Backend || b.Backend == c.Backend || a.Backend == c.Backend {
		t.Fatal("three requests on an idle pool should land on three backends")
	}
	b.Release()
	if got := mustAcquire(t, p, Request{}).Backend; got != b.Backend {
		t.Errorf("got %s, want the freed backend %s", got.Name, b.Backend.Name)
	}
}

func TestPrefixAwareKeepsAffinity(t *testing.T) {
	p := NewPool(newBackends(3, 0), &PrefixAware{MinMatchRatio: 0.5, LoadSlack: 2}, Thresholds{})
	docA := strings.Repeat("document A ", 20)
	docB := strings.Repeat("document B ", 20)

	first := mustAcquire(t, p, Request{Hashes: hashes(docA + "question 1")})
	first.Release()
	if first.Reason != "least_loaded" {
		t.Errorf("first request: reason %q, want least_loaded", first.Reason)
	}
	for i := range 5 {
		l := mustAcquire(t, p, Request{Hashes: hashes(docA + fmt.Sprintf("question %d", i+2))})
		l.Release()
		if l.Backend != first.Backend || l.Reason != "prefix" {
			t.Fatalf("request %d went to %s (%s), want %s (prefix)", i, l.Backend.Name, l.Reason, first.Backend.Name)
		}
		if l.Matched == 0 {
			t.Error("Matched should count the shared document")
		}
	}

	// A different document is new: it goes by load, which here rotates it
	// away from the backend that holds document A.
	other := mustAcquire(t, p, Request{Hashes: hashes(docB + "question")})
	if other.Reason != "least_loaded" {
		t.Errorf("new document: reason %q, want least_loaded", other.Reason)
	}
}

func TestPrefixAwareLoadSlackBreaksAffinity(t *testing.T) {
	p := NewPool(newBackends(2, 0), &PrefixAware{MinMatchRatio: 0.5, LoadSlack: 1}, Thresholds{})
	doc := strings.Repeat("shared context ", 20)

	// Two requests stay on the cached backend: the second runs one more
	// request than the idle backend, which the slack allows.
	a := mustAcquire(t, p, Request{Hashes: hashes(doc + "q1")})
	b := mustAcquire(t, p, Request{Hashes: hashes(doc + "q2")})
	if b.Backend != a.Backend {
		t.Fatal("second request should follow the cache within the slack")
	}
	// The third would make the gap 2 > slack, so it goes to the idle backend.
	c := mustAcquire(t, p, Request{Hashes: hashes(doc + "q3")})
	if c.Backend == a.Backend || c.Reason != "least_loaded" {
		t.Errorf("third request went to %s (%s), want the other backend by load", c.Backend.Name, c.Reason)
	}
}

func TestPrefixAwareSpreadsSharedPrefixByLoad(t *testing.T) {
	bs := newBackends(3, 0)
	p := NewPool(bs, &PrefixAware{MinMatchRatio: 0.5, LoadSlack: 10}, Thresholds{})
	system := strings.Repeat("You are a support assistant. ", 10)
	// Every backend has served the shared system prompt before.
	for _, b := range bs {
		b.index.Insert(hashes(system))
	}

	// New chats match all three equally, so they go by load instead of
	// piling onto the first backend in the list.
	seen := map[*Backend]bool{}
	for i := range 3 {
		// The first messages differ from their first byte, so no backend
		// holds more than the system prompt.
		l := mustAcquire(t, p, Request{Hashes: hashes(system + fmt.Sprintf("%d: hello there", i))})
		if l.Reason != "prefix" {
			t.Errorf("reason %q, want prefix: the system prompt is cached", l.Reason)
		}
		seen[l.Backend] = true
	}
	if len(seen) != 3 {
		t.Errorf("three new chats landed on %d backends, want 3", len(seen))
	}
}

func TestPrefixAwareMinMatchRatio(t *testing.T) {
	p := NewPool(newBackends(2, 0), &PrefixAware{MinMatchRatio: 0.5, LoadSlack: 10}, Thresholds{})
	shortShared := "system: be nice. "
	l := mustAcquire(t, p, Request{Hashes: hashes(shortShared + strings.Repeat("x", 100))})
	l.Release()

	// Only the short system prompt is shared, far below half of the prompt.
	l2 := mustAcquire(t, p, Request{Hashes: hashes(shortShared + strings.Repeat("y", 100))})
	if l2.Reason != "least_loaded" {
		t.Errorf("reason %q, want least_loaded for a small shared prefix", l2.Reason)
	}
}

func TestAcquireWaitsForFreeSlot(t *testing.T) {
	p := NewPool(newBackends(1, 1), &LeastLoaded{}, Thresholds{})
	held := mustAcquire(t, p, Request{})

	got := make(chan *Lease)
	go func() {
		l, err := p.Acquire(context.Background(), Request{})
		if err != nil {
			t.Error(err)
		}
		got <- l
	}()

	select {
	case <-got:
		t.Fatal("Acquire returned while the only slot was taken")
	case <-time.After(20 * time.Millisecond):
	}
	held.Release()
	select {
	case l := <-got:
		if l.Backend.Inflight() != 1 {
			t.Errorf("inflight = %d, want 1", l.Backend.Inflight())
		}
	case <-time.After(time.Second):
		t.Fatal("waiting request did not get the freed slot")
	}
}

func TestAcquireTimesOutWhenFull(t *testing.T) {
	p := NewPool(newBackends(1, 1), &LeastLoaded{}, Thresholds{})
	mustAcquire(t, p, Request{})

	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	if _, err := p.Acquire(ctx, Request{}); !errors.Is(err, context.DeadlineExceeded) {
		t.Errorf("err = %v, want deadline exceeded", err)
	}
}

func TestReleaseIsIdempotent(t *testing.T) {
	p := NewPool(newBackends(1, 0), &LeastLoaded{}, Thresholds{})
	l := mustAcquire(t, p, Request{})
	l.Release()
	l.Release()
	if n := l.Backend.Inflight(); n != 0 {
		t.Errorf("inflight = %d after double release, want 0", n)
	}
}

func TestExcludeAndNoBackend(t *testing.T) {
	bs := newBackends(2, 0)
	p := NewPool(bs, &RoundRobin{}, Thresholds{})
	for range 4 {
		if l := mustAcquire(t, p, Request{Exclude: []*Backend{bs[0]}}); l.Backend != bs[1] {
			t.Fatal("excluded backend was picked")
		}
	}
	if _, err := p.Acquire(context.Background(), Request{Exclude: bs}); !errors.Is(err, ErrNoBackend) {
		t.Errorf("err = %v, want ErrNoBackend", err)
	}
}

func TestObserveThresholds(t *testing.T) {
	bs := newBackends(1, 0)
	b := bs[0]
	p := NewPool(bs, &LeastLoaded{}, Thresholds{Fall: 2, Rise: 2})
	b.index.Insert([]uint64{1, 2, 3})

	if p.Observe(b, false) || !b.Healthy() {
		t.Fatal("one failure must not eject with Fall=2")
	}
	if !p.Observe(b, false) || b.Healthy() {
		t.Fatal("second failure should eject")
	}
	if _, err := p.Acquire(context.Background(), Request{}); !errors.Is(err, ErrNoBackend) {
		t.Errorf("err = %v, want ErrNoBackend while the only backend is down", err)
	}
	if p.Observe(b, true) || b.Healthy() {
		t.Fatal("one success must not readmit with Rise=2")
	}
	if !p.Observe(b, true) || !b.Healthy() {
		t.Fatal("second success should readmit")
	}
	if b.index.Len() != 0 {
		t.Error("a backend that came back should start with an empty cache index")
	}
}

func TestEjectWakesWaiters(t *testing.T) {
	bs := newBackends(1, 1)
	p := NewPool(bs, &LeastLoaded{}, Thresholds{Fall: 1, Rise: 1})
	mustAcquire(t, p, Request{})

	errc := make(chan error)
	go func() {
		_, err := p.Acquire(context.Background(), Request{})
		errc <- err
	}()
	time.Sleep(10 * time.Millisecond)
	p.Observe(bs[0], false)
	select {
	case err := <-errc:
		if !errors.Is(err, ErrNoBackend) {
			t.Errorf("err = %v, want ErrNoBackend", err)
		}
	case <-time.After(time.Second):
		t.Fatal("waiter was not told that the last backend went down")
	}
}

func TestNewPolicy(t *testing.T) {
	for _, name := range []string{"round_robin", "least_loaded", "prefix"} {
		p, err := NewPolicy(name, 0.5, 2)
		if err != nil || p.Name() != name {
			t.Errorf("NewPolicy(%q) = %v, %v", name, p, err)
		}
	}
	if _, err := NewPolicy("random", 0, 0); err == nil {
		t.Error("unknown policy should fail")
	}
}
