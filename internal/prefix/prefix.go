// Package prefix tracks which prompt prefixes each backend has recently
// served, so the router can send a request to the backend most likely to
// still hold its KV cache.
package prefix

import (
	"container/list"
	"hash/fnv"
	"io"
	"sync"
)

// Hashes splits text into chunkSize-byte chunks and returns one hash per
// chunk. Each hash covers the whole prefix up to and including its chunk, so
// two prompts share hash i only if their first i+1 chunks are equal. A
// trailing partial chunk is dropped: backends reuse cache in whole blocks.
func Hashes(text string, chunkSize int) []uint64 {
	if chunkSize <= 0 {
		return nil
	}
	n := len(text) / chunkSize
	if n == 0 {
		return nil
	}
	out := make([]uint64, n)
	h := fnv.New64a()
	for i := range n {
		_, _ = io.WriteString(h, text[i*chunkSize:(i+1)*chunkSize])
		out[i] = h.Sum64()
	}
	return out
}

// Index is a fixed-capacity LRU set of prefix hashes for one backend. It
// approximates the backend's KV cache: prefixes routed there recently are
// assumed to still be cached.
type Index struct {
	mu       sync.Mutex
	capacity int
	ll       *list.List // front is most recently used
	items    map[uint64]*list.Element
}

// NewIndex returns an empty index that holds at most capacity hashes.
func NewIndex(capacity int) *Index {
	return &Index{
		capacity: max(capacity, 1),
		ll:       list.New(),
		items:    make(map[uint64]*list.Element),
	}
}

// Match returns how many leading hashes of a prompt the index holds.
func (x *Index) Match(hashes []uint64) int {
	x.mu.Lock()
	defer x.mu.Unlock()
	for i, h := range hashes {
		if _, ok := x.items[h]; !ok {
			return i
		}
	}
	return len(hashes)
}

// Insert records a prompt as served. Hashes are touched from last to first,
// so the shortest prefixes end up most recent and are evicted last: a shared
// system prompt outlives the individual questions that follow it, and the
// index never keeps a long prefix while dropping its start.
func (x *Index) Insert(hashes []uint64) {
	x.mu.Lock()
	defer x.mu.Unlock()
	for i := len(hashes) - 1; i >= 0; i-- {
		h := hashes[i]
		if e, ok := x.items[h]; ok {
			x.ll.MoveToFront(e)
			continue
		}
		x.items[h] = x.ll.PushFront(h)
	}
	for x.ll.Len() > x.capacity {
		e := x.ll.Back()
		x.ll.Remove(e)
		delete(x.items, e.Value.(uint64))
	}
}

// Reset forgets everything, for example after the backend restarted and lost
// its cache.
func (x *Index) Reset() {
	x.mu.Lock()
	defer x.mu.Unlock()
	x.ll.Init()
	clear(x.items)
}

// Len returns the number of hashes held.
func (x *Index) Len() int {
	x.mu.Lock()
	defer x.mu.Unlock()
	return x.ll.Len()
}
