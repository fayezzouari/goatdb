package index

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"

	"github.com/fayezzouari/goatdb/core"
)

// efRecall returns the mean recall@topK of SearchEf against exact search.
func efRecall(idx core.EfSearcher, flat *FlatIndex, queries []core.Vector, topK, ef int) float64 {
	var total float64
	for _, q := range queries {
		gt := make(map[string]struct{}, topK)
		for _, r := range flat.Search(q, topK) {
			gt[r.Id] = struct{}{}
		}
		hits := 0
		for _, r := range idx.SearchEf(q, topK, ef) {
			if _, ok := gt[r.Id]; ok {
				hits++
			}
		}
		total += float64(hits) / float64(len(gt))
	}
	return total / float64(len(queries))
}

func TestHNSWSearchEfRecallMonotonic(t *testing.T) {
	const n, dim, topK = 3000, 32, 10
	r := rand.New(rand.NewSource(1))
	// A small M and efConstruction make a weak graph, so ef matters.
	h := NewHNSWIndex(dim, 4, 16, 10, core.Euclidean)
	flat := NewFlatIndex(dim, core.Euclidean)
	for i, v := range randomVectors(r, n, dim) {
		id := fmt.Sprintf("v%d", i)
		h.AddVector(id, v)
		flat.AddVector(id, v)
	}
	queries := randomVectors(r, 100, dim)

	low := efRecall(h, flat, queries, topK, topK)
	high := efRecall(h, flat, queries, topK, 400)
	t.Logf("recall@%d ef=%d: %.3f, ef=400: %.3f", topK, topK, low, high)
	if high < low {
		t.Errorf("recall with ef=400 (%.3f) is below recall with ef=%d (%.3f)", high, topK, low)
	}
	if high < 0.9 {
		t.Errorf("recall with ef=400 = %.3f, want >= 0.9", high)
	}
}

func TestHNSWSearchEfBelowTopK(t *testing.T) {
	const n, dim, topK = 500, 8, 50
	r := rand.New(rand.NewSource(2))
	h := NewHNSWIndex(dim, 16, 100, 64, core.Euclidean)
	for i, v := range randomVectors(r, n, dim) {
		h.AddVector(fmt.Sprintf("v%d", i), v)
	}
	q := randomVectors(r, 1, dim)[0]
	// ef < topK must still return topK results, sorted by distance.
	res := h.SearchEf(q, topK, 1)
	if len(res) != topK {
		t.Fatalf("got %d results, want %d", len(res), topK)
	}
	for i := 1; i < len(res); i++ {
		if res[i].Distance < res[i-1].Distance {
			t.Fatalf("results not sorted at %d", i)
		}
	}
}

func TestIVFSearchEfRecallMonotonic(t *testing.T) {
	const n, dim, topK = 3000, 16, 10
	r := rand.New(rand.NewSource(3))
	vecs := randomVectors(r, n, dim)
	idx := NewIVFIndex(dim, 50, 1, core.Euclidean)
	idx.Train(vecs)
	flat := NewFlatIndex(dim, core.Euclidean)
	for i, v := range vecs {
		id := fmt.Sprintf("v%d", i)
		idx.AddVector(id, v)
		flat.AddVector(id, v)
	}
	queries := randomVectors(r, 100, dim)

	low := efRecall(idx, flat, queries, topK, 1)
	high := efRecall(idx, flat, queries, topK, 20)
	all := efRecall(idx, flat, queries, topK, 1000) // more than nlist: probes every list
	t.Logf("recall@%d nprobe=1: %.3f, nprobe=20: %.3f, nprobe=1000: %.3f", topK, low, high, all)
	if high < low {
		t.Errorf("recall with nprobe=20 (%.3f) is below recall with nprobe=1 (%.3f)", high, low)
	}
	if all < 0.999 {
		t.Errorf("probing every list gave recall %.3f, want 1", all)
	}
	if got := len(idx.SearchEf(queries[0], topK, 0)); got != topK {
		t.Errorf("SearchEf with ef=0 returned %d results, want %d", got, topK)
	}
}

// TestSearchEfConcurrent runs searches with different ef in parallel; run
// with -race to check that SearchEf does not mutate shared state.
func TestSearchEfConcurrent(t *testing.T) {
	const n, dim = 1000, 8
	r := rand.New(rand.NewSource(4))
	vecs := randomVectors(r, n, dim)
	h := NewHNSWIndex(dim, 8, 50, 20, core.Euclidean)
	ivf := NewIVFIndex(dim, 10, 2, core.Euclidean)
	ivf.Train(vecs)
	for i, v := range vecs {
		id := fmt.Sprintf("v%d", i)
		h.AddVector(id, v)
		ivf.AddVector(id, v)
	}
	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(ef int) {
			defer wg.Done()
			for i := 0; i < 50; i++ {
				q := vecs[(i*7+ef)%n]
				h.SearchEf(q, 5, ef)
				ivf.SearchEf(q, 5, ef)
			}
		}(g*10 + 1)
	}
	wg.Wait()
	if h.ef != 20 || ivf.nProbe != 2 {
		t.Errorf("configured ef/nprobe changed: %d/%d", h.ef, ivf.nProbe)
	}
}
