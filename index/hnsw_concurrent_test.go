package index

import (
	"fmt"
	"math/rand"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/fayezzouari/goatdb/core"
)

// buildParallel inserts vecs with the given number of goroutines.
func buildParallel(h *HNSWIndex, vecs []core.Vector, workers int) {
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1) - 1)
				if i >= len(vecs) {
					return
				}
				h.AddVector(fmt.Sprintf("v%d", i), vecs[i])
			}
		}()
	}
	wg.Wait()
}

// TestHNSWParallelBuildRecall checks that a graph built by concurrent
// inserts is as good as a serially built one.
func TestHNSWParallelBuildRecall(t *testing.T) {
	n, nq := 8000, 200
	if testing.Short() || raceEnabled {
		n, nq = 3000, 100
	}
	const dim, topK, ef = 24, 10, 32
	r := rand.New(rand.NewSource(42))
	vecs := randVecs(r, n, dim)
	queries := randVecs(r, nq, dim)
	flat := NewFlatIndex(dim, core.Euclidean)
	for i, v := range vecs {
		flat.AddVector(fmt.Sprintf("v%d", i), v)
	}

	serial := NewHNSWIndex(dim, 12, 100, ef, core.Euclidean)
	buildParallel(serial, vecs, 1)
	parallel := NewHNSWIndex(dim, 12, 100, ef, core.Euclidean)
	buildParallel(parallel, vecs, 8)
	checkGraph(t, parallel)

	rs := efRecall(serial, flat, queries, topK, ef)
	rp := efRecall(parallel, flat, queries, topK, ef)
	t.Logf("recall@%d ef=%d: serial %.4f parallel %.4f", topK, ef, rs, rp)
	if rp < rs-0.02 {
		t.Errorf("parallel build recall %.4f is well below serial %.4f", rp, rs)
	}
}

// TestHNSWConcurrentStress runs inserts, searches, deletes, GetVector and
// Train together; run it with -race. Deleted ids must never come back, and
// every id that was inserted and not deleted must be retrievable.
func TestHNSWConcurrentStress(t *testing.T) {
	const dim = 12
	perWriter := 600
	if testing.Short() || raceEnabled {
		perWriter = 250
	}
	h := NewHNSWIndex(dim, 8, 40, 32, core.Euclidean)
	seed := rand.New(rand.NewSource(1))
	trainSet := randVecs(seed, 300, dim)
	for i, v := range trainSet {
		h.AddVector(fmt.Sprintf("seed%d", i), v)
	}

	const writers = 4
	var deleted sync.Map
	var wg sync.WaitGroup
	stop := make(chan struct{})
	errs := make(chan string, 16)
	report := func(format string, args ...any) {
		select {
		case errs <- fmt.Sprintf(format, args...):
		default:
		}
	}

	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(100 + w)))
			for i := 0; i < perWriter; i++ {
				id := fmt.Sprintf("w%d-%d", w, i)
				h.AddVector(id, randVecs(r, 1, dim)[0])
				// Delete some of this writer's earlier ids, so inserts
				// recycle free slots while others are running.
				if i%5 == 4 {
					victim := fmt.Sprintf("w%d-%d", w, i-2)
					if !h.DeleteVector(victim) {
						report("delete %s failed", victim)
					}
					deleted.Store(victim, true)
				}
			}
		}(w)
	}
	var readers sync.WaitGroup
	for g := 0; g < 4; g++ {
		readers.Add(1)
		go func(g int) {
			defer readers.Done()
			r := rand.New(rand.NewSource(int64(200 + g)))
			for {
				select {
				case <-stop:
					return
				default:
				}
				q := randVecs(r, 1, dim)[0]
				res := h.SearchEf(q, 5, 32)
				if len(res) == 0 {
					report("search returned no results")
					return
				}
				seen := map[string]bool{}
				for _, x := range res {
					if x.Id == "" || seen[x.Id] {
						report("bad result set %v", res)
						return
					}
					seen[x.Id] = true
				}
				h.GetVector(fmt.Sprintf("w%d-%d", g, r.Intn(perWriter)))
			}
		}(g)
	}
	readers.Add(1)
	go func() {
		defer readers.Done()
		for i := 0; i < 3; i++ {
			h.Train(trainSet)
		}
	}()

	wg.Wait()
	close(stop)
	readers.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}

	checkGraph(t, h)
	checkSQCodes(t, h)
	for w := 0; w < writers; w++ {
		for i := 0; i < perWriter; i++ {
			id := fmt.Sprintf("w%d-%d", w, i)
			_, gone := deleted.Load(id)
			if _, ok := h.GetVector(id); ok == gone {
				t.Fatalf("%s: present=%v deleted=%v", id, ok, gone)
			}
		}
	}
	// Every live vector is still its own nearest neighbor.
	h.mu.RLock()
	ids := make(map[string]int32, len(h.ids))
	for id, slot := range h.ids {
		ids[id] = slot
	}
	h.mu.RUnlock()
	misses := 0
	for id := range ids {
		v, _ := h.GetVector(id)
		if res := h.SearchEf(v, 1, 64); len(res) != 1 || res[0].Id != id {
			misses++
		}
	}
	if misses > len(ids)/100 {
		t.Errorf("%d of %d vectors are not found by their own query", misses, len(ids))
	}
}

// TestHNSWSaveDuringInserts checks that Save writes a consistent graph
// while inserts are running.
func TestHNSWSaveDuringInserts(t *testing.T) {
	const dim = 8
	r := rand.New(rand.NewSource(3))
	h := NewHNSWIndex(dim, 6, 30, 30, core.Cosine)
	vecs := randVecs(r, 2000, dim)
	path := t.TempDir() + "/hnsw.bin"
	done := make(chan struct{})
	go func() {
		buildParallel(h, vecs, 4)
		close(done)
	}()
	for i := 0; i < 3; i++ {
		if err := h.Save(path); err != nil {
			t.Fatal(err)
		}
		h2 := NewHNSWIndex(dim, 6, 30, 30, core.Cosine)
		if err := h2.Load(path); err != nil {
			t.Fatal(err)
		}
		checkGraph(t, h2)
	}
	<-done
}
