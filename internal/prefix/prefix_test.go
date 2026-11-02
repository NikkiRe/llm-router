package prefix

import (
	"strings"
	"testing"
)

func TestHashesSharePrefix(t *testing.T) {
	system := strings.Repeat("s", 40)
	a := Hashes(system+"question one", 10)
	b := Hashes(system+"another question", 10)

	if len(a) != 5 || len(b) != 5 {
		t.Fatalf("got %d and %d chunks, want 5 and 5", len(a), len(b))
	}
	for i := range 4 {
		if a[i] != b[i] {
			t.Errorf("chunk %d differs although the first 40 bytes are equal", i)
		}
	}
	if a[4] == b[4] {
		t.Error("chunk 4 is equal although the prompts differ there")
	}
}

func TestHashesDependOnWholePrefix(t *testing.T) {
	// The second chunk is the same text, but it follows a different first
	// chunk, so the backend cannot reuse it.
	a := Hashes("aaaaXXXX", 4)
	b := Hashes("bbbbXXXX", 4)
	if a[1] == b[1] {
		t.Error("equal chunks after different prefixes must hash differently")
	}
}

func TestHashesDropPartialChunk(t *testing.T) {
	if got := len(Hashes("abcdefg", 4)); got != 1 {
		t.Errorf("got %d chunks, want 1", got)
	}
	if got := Hashes("abc", 4); got != nil {
		t.Errorf("got %v for a prompt shorter than one chunk, want nil", got)
	}
	if got := Hashes("abc", 0); got != nil {
		t.Errorf("got %v for chunk size 0, want nil", got)
	}
}

func TestIndexMatchStopsAtFirstMiss(t *testing.T) {
	x := NewIndex(100)
	x.Insert([]uint64{1, 2, 3})

	cases := []struct {
		hashes []uint64
		want   int
	}{
		{[]uint64{1, 2, 3, 4}, 3},
		{[]uint64{1, 2}, 2},
		{[]uint64{1, 9, 3}, 1},
		{[]uint64{9}, 0},
		{nil, 0},
	}
	for _, c := range cases {
		if got := x.Match(c.hashes); got != c.want {
			t.Errorf("Match(%v) = %d, want %d", c.hashes, got, c.want)
		}
	}
}

func TestIndexEvictsLongestPrefixesFirst(t *testing.T) {
	x := NewIndex(4)
	x.Insert([]uint64{1, 2, 3}) // shared system prompt
	x.Insert([]uint64{1, 2, 3, 4, 5})

	// Capacity 4 forces one eviction. The tail of the latest prompt goes,
	// the shared start stays.
	if got := x.Match([]uint64{1, 2, 3, 4, 5}); got != 4 {
		t.Errorf("Match = %d, want 4", got)
	}
	if x.Len() != 4 {
		t.Errorf("Len = %d, want 4", x.Len())
	}
}

func TestIndexKeepsRecentlyUsedPrompt(t *testing.T) {
	x := NewIndex(4)
	x.Insert([]uint64{1, 2})
	x.Insert([]uint64{10, 20})
	x.Insert([]uint64{1, 2}) // used again, becomes most recent
	x.Insert([]uint64{30, 40})

	if got := x.Match([]uint64{1, 2}); got != 2 {
		t.Errorf("recently used prompt: Match = %d, want 2", got)
	}
	if got := x.Match([]uint64{10, 20}); got != 0 {
		t.Errorf("least recently used prompt: Match = %d, want 0", got)
	}
}

func TestIndexReset(t *testing.T) {
	x := NewIndex(10)
	x.Insert([]uint64{1, 2, 3})
	x.Reset()
	if x.Len() != 0 || x.Match([]uint64{1}) != 0 {
		t.Error("index not empty after Reset")
	}
}

func BenchmarkMatch(b *testing.B) {
	x := NewIndex(1 << 16)
	text := strings.Repeat("You are a helpful assistant. ", 400)
	hashes := Hashes(text, 256)
	x.Insert(hashes)
	b.ResetTimer()
	for b.Loop() {
		x.Match(hashes)
	}
}
