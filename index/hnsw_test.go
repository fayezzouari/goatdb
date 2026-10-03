package index

import (
	"encoding/gob"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sync"
	"testing"

	"github.com/fayez/goatdb/core"
)

func TestHNSWIndexSearch(t *testing.T) {
	idx := NewHNSWIndex(3, 4, 20, 10, core.Euclidean)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0, 0}})
	idx.AddVector("b", core.Vector{Embeddings: []float32{0, 1, 0}})
	idx.AddVector("c", core.Vector{Embeddings: []float32{0, 0, 1}})

	results := idx.Search(core.Vector{Embeddings: []float32{1, 0, 0}}, 1)
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].Id != "a" {
		t.Errorf("expected 'a', got %q", results[0].Id)
	}
}

func TestHNSWIndexTopK(t *testing.T) {
	idx := NewHNSWIndex(2, 4, 20, 10, core.Euclidean)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})
	idx.AddVector("b", core.Vector{Embeddings: []float32{2, 0}})
	idx.AddVector("c", core.Vector{Embeddings: []float32{10, 0}})

	results := idx.Search(core.Vector{Embeddings: []float32{0, 0}}, 2)
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	ids := map[string]bool{results[0].Id: true, results[1].Id: true}
	if !ids["a"] || !ids["b"] {
		t.Errorf("expected top-2 to be 'a' and 'b', got %v", results)
	}
}

func TestHNSWIndexGetVector(t *testing.T) {
	idx := NewHNSWIndex(2, 4, 20, 10, core.Euclidean)
	idx.AddVector("x", core.Vector{Embeddings: []float32{3, 4}})

	v, ok := idx.GetVector("x")
	if !ok {
		t.Fatal("expected vector 'x'")
	}
	if v.Embeddings[0] != 3 || v.Embeddings[1] != 4 {
		t.Errorf("wrong embeddings: %v", v.Embeddings)
	}
}

func TestHNSWIndexDelete(t *testing.T) {
	idx := NewHNSWIndex(2, 4, 20, 10, core.Euclidean)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})
	idx.AddVector("b", core.Vector{Embeddings: []float32{0, 1}})

	if ok := idx.DeleteVector("a"); !ok {
		t.Fatal("expected delete to return true")
	}
	if _, ok := idx.GetVector("a"); ok {
		t.Error("expected 'a' to be deleted")
	}
	if idx.DeleteVector("a") {
		t.Error("expected second delete to return false")
	}

	// Index should still be searchable after delete
	results := idx.Search(core.Vector{Embeddings: []float32{0, 1}}, 1)
	if len(results) == 0 || results[0].Id != "b" {
		t.Errorf("expected 'b' after deleting 'a', got %v", results)
	}
}

func TestHNSWIndexSearchEmpty(t *testing.T) {
	idx := NewHNSWIndex(2, 4, 20, 10, core.Euclidean)
	results := idx.Search(core.Vector{Embeddings: []float32{1, 0}}, 5)
	if results != nil {
		t.Errorf("expected nil from empty index, got %v", results)
	}
}

func TestHNSWIndexSaveLoad(t *testing.T) {
	path := t.TempDir() + "/hnsw.bin"

	idx := NewHNSWIndex(2, 4, 20, 10, core.Cosine)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})
	idx.AddVector("b", core.Vector{Embeddings: []float32{0, 1}})

	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}

	idx2 := NewHNSWIndex(2, 4, 20, 10, core.Cosine)
	if err := idx2.Load(path); err != nil {
		t.Fatal(err)
	}
	if _, ok := idx2.GetVector("a"); !ok {
		t.Error("vector 'a' not found after load")
	}
	results := idx2.Search(core.Vector{Embeddings: []float32{1, 0}}, 1)
	if len(results) == 0 || results[0].Id != "a" {
		t.Errorf("expected 'a' after load, got %v", results)
	}
}

func TestHNSWRecallWithCodebook(t *testing.T) {
	const n, dim, topK = 1000, 128, 10
	rand.Seed(42)

	vecs := make([]core.Vector, n)
	for i := range vecs {
		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = rand.Float32()*2 - 1
		}
		vecs[i] = core.Vector{Embeddings: emb}
	}

	// flat ground truth
	flat := NewFlatIndex(dim, core.Euclidean)
	for i, v := range vecs {
		flat.AddVector(fmt.Sprintf("v%d", i), v)
	}

	// HNSW with Train-before-AddVector (mirrors benchmark)
	hnsw := NewHNSWIndex(dim, 16, 200, 128, core.Euclidean)
	hnsw.Train(vecs)
	for i, v := range vecs {
		hnsw.AddVector(fmt.Sprintf("v%d", i), v)
	}

	query := core.Vector{Embeddings: vecs[0].Embeddings}
	gtResults := flat.Search(query, topK)
	gt := make(map[string]struct{}, len(gtResults))
	for _, r := range gtResults {
		gt[r.Id] = struct{}{}
	}

	hnswResults := hnsw.Search(query, topK)
	hits := 0
	for _, r := range hnswResults {
		if _, ok := gt[r.Id]; ok {
			hits++
		}
	}
	recall := float64(hits) / float64(len(gt))
	t.Logf("recall@%d = %.1f%% (%d/%d)", topK, recall*100, hits, len(gt))
	if recall < 0.5 {
		t.Errorf("recall too low: %.1f%% (expected >= 50%%)", recall*100)
	}
}

func TestHNSWRecallNoCodebook(t *testing.T) {
	const n, dim, topK = 1000, 128, 10
	rand.Seed(42)

	vecs := make([]core.Vector, n)
	for i := range vecs {
		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = rand.Float32()*2 - 1
		}
		vecs[i] = core.Vector{Embeddings: emb}
	}

	flat := NewFlatIndex(dim, core.Euclidean)
	for i, v := range vecs {
		flat.AddVector(fmt.Sprintf("v%d", i), v)
	}

	// HNSW WITHOUT Train (no codebook)
	hnsw := NewHNSWIndex(dim, 16, 200, 128, core.Euclidean)
	for i, v := range vecs {
		hnsw.AddVector(fmt.Sprintf("v%d", i), v)
	}

	var totalRecall float64
	const nq = 50
	for qi := 0; qi < nq; qi++ {
		query := vecs[qi]
		gtResults := flat.Search(query, topK)
		gt := make(map[string]struct{}, len(gtResults))
		for _, r := range gtResults {
			gt[r.Id] = struct{}{}
		}
		hnswResults := hnsw.Search(query, topK)
		hits := 0
		for _, r := range hnswResults {
			if _, ok := gt[r.Id]; ok {
				hits++
			}
		}
		totalRecall += float64(hits) / float64(len(gt))
	}
	recall := totalRecall / nq
	t.Logf("avg recall@%d (no codebook) = %.1f%%", topK, recall*100)
	if recall < 0.5 {
		t.Errorf("recall too low: %.1f%%", recall*100)
	}
}

// clusteredVectors returns n vectors drawn from nClusters tight gaussian
// clusters whose centers are spread across [-10, 10]^dim.
func clusteredVectors(r *rand.Rand, n, nClusters, dim int) []core.Vector {
	centers := make([][]float32, nClusters)
	for c := range centers {
		centers[c] = make([]float32, dim)
		for j := range centers[c] {
			centers[c][j] = r.Float32()*20 - 10
		}
	}
	vecs := make([]core.Vector, n)
	for i := range vecs {
		center := centers[r.Intn(nClusters)]
		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = center[j] + float32(r.NormFloat64())*0.5
		}
		vecs[i] = core.Vector{Embeddings: emb}
	}
	return vecs
}

// avgRecall returns the mean recall@topK of idx against exact search.
func avgRecall(idx core.Index, flat *FlatIndex, queries []core.Vector, topK int) float64 {
	var total float64
	for _, q := range queries {
		gt := make(map[string]struct{}, topK)
		for _, r := range flat.Search(q, topK) {
			gt[r.Id] = struct{}{}
		}
		hits := 0
		for _, r := range idx.Search(q, topK) {
			if _, ok := gt[r.Id]; ok {
				hits++
			}
		}
		total += float64(hits) / float64(len(gt))
	}
	return total / float64(len(queries))
}

// TestHNSWRecallClustered measures recall on clustered data, where naive
// closest-M neighbor selection links every node into its own cluster only.
func TestHNSWRecallClustered(t *testing.T) {
	const n, nClusters, dim, topK, nq = 10000, 200, 32, 10, 200
	r := rand.New(rand.NewSource(11))
	vecs := clusteredVectors(r, n+nq, nClusters, dim)
	data, queries := vecs[:n], vecs[n:]

	flat := NewFlatIndex(dim, core.Euclidean)
	hnsw := NewHNSWIndex(dim, 8, 64, 10, core.Euclidean)
	for i, v := range data {
		id := fmt.Sprintf("v%d", i)
		flat.AddVector(id, v)
		hnsw.AddVector(id, v)
	}

	var recall float64
	for _, ef := range []int{10, 20, 40, 80} {
		hnsw.ef = ef
		recall = avgRecall(hnsw, flat, queries, topK)
		t.Logf("clustered recall@%d ef=%d: %.1f%%", topK, ef, recall*100)
	}
	if recall < 0.9 {
		t.Errorf("recall at ef=80 too low: %.1f%% (expected >= 90%%)", recall*100)
	}
}

// checkSQCodes verifies that every indexed vector's int8 code matches the
// current codebook.
func checkSQCodes(t *testing.T, h *HNSWIndex) {
	t.Helper()
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.codebook == nil {
		t.Fatal("expected a codebook after Train")
	}
	for id, node := range h.nodes {
		want := h.codebook.Quantize(h.pool.Get(node.poolIdx))
		got := h.sqPool.Get(node.poolIdx)
		for i := range want {
			if got[i] != want[i] {
				t.Fatalf("vector %s: stale int8 code at dim %d", id, i)
			}
		}
	}
}

// TestHNSWSearchDuringTrain checks that Search and AddVector run while Train
// is quantizing, and that vectors added meanwhile get the new codebook.
func TestHNSWSearchDuringTrain(t *testing.T) {
	const dim = 16
	r := rand.New(rand.NewSource(9))
	vecs := make([]core.Vector, trainChunk+100)
	for i := range vecs {
		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = r.Float32()
		}
		vecs[i] = core.Vector{Embeddings: emb}
	}
	h := NewHNSWIndex(dim, 8, 32, 16, core.Euclidean)
	for i, v := range vecs {
		h.AddVector(fmt.Sprintf("v%d", i), v)
	}

	paused := make(chan struct{})
	resume := make(chan struct{})
	var once sync.Once
	testHookTrainChunk = func() {
		once.Do(func() {
			close(paused)
			<-resume
		})
	}
	t.Cleanup(func() { testHookTrainChunk = nil })

	done := make(chan struct{})
	go func() {
		h.Train(vecs)
		close(done)
	}()

	<-paused
	// Train is mid-way through quantizing; neither call may block.
	if res := h.Search(vecs[0], 1); len(res) != 1 || res[0].Id != "v0" {
		t.Errorf("search during Train: got %v", res)
	}
	h.AddVector("during", core.Vector{Embeddings: vecs[1].Embeddings})
	close(resume)
	<-done

	checkSQCodes(t, h)
	if res := h.Search(vecs[2], 1); len(res) != 1 || res[0].Id != "v2" {
		t.Errorf("search after Train: got %v", res)
	}
}

// TestHNSWConcurrentTrainAddSearch runs Train, AddVector and Search together;
// run it with -race.
func TestHNSWConcurrentTrainAddSearch(t *testing.T) {
	const dim = 16
	r := rand.New(rand.NewSource(13))
	vec := func(r *rand.Rand) core.Vector {
		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = r.Float32()
		}
		return core.Vector{Embeddings: emb}
	}
	h := NewHNSWIndex(dim, 8, 32, 16, core.Euclidean)
	train := make([]core.Vector, 500)
	for i := range train {
		train[i] = vec(r)
		h.AddVector(fmt.Sprintf("seed%d", i), train[i])
	}

	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		for i := 0; i < 5; i++ {
			h.Train(train)
		}
	}()
	go func() {
		defer wg.Done()
		r := rand.New(rand.NewSource(14))
		for i := 0; i < 300; i++ {
			h.AddVector(fmt.Sprintf("w%d", i), vec(r))
		}
	}()
	go func() {
		defer wg.Done()
		r := rand.New(rand.NewSource(15))
		for i := 0; i < 300; i++ {
			if res := h.Search(vec(r), 5); len(res) != 5 {
				t.Errorf("expected 5 results, got %d", len(res))
				return
			}
		}
	}()
	wg.Wait()
	checkSQCodes(t, h)
}

// TestHNSWLoadLegacyPQ checks that files written with PQ data still load.
func TestHNSWLoadLegacyPQ(t *testing.T) {
	path := t.TempDir() + "/hnsw.bin"
	state := hnswState{
		Dim: 2, M: 4, EfConstruction: 20, Ef: 10, ML: 1 / math.Log(4),
		DistanceMetric: core.Euclidean,
		EntryPoint:     "a",
		MaxLayer:       0,
		Nodes: map[string]hnswNodeState{
			"a": {Embeddings: []float32{1, 0}, Connections: [][]string{{"b"}}},
			"b": {Embeddings: []float32{0, 1}, Connections: [][]string{{"a"}}},
		},
		SQMin: 0, SQScale: 1.0 / 255, HasSQ: true,
		PQNSubs: 1, PQNCentroids: 2, PQCentroids: []float32{1, 0, 0, 1}, HasPQ: true,
	}
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gob.NewEncoder(f).Encode(state); err != nil {
		t.Fatal(err)
	}
	f.Close()

	h := NewHNSWIndex(2, 4, 20, 10, core.Euclidean)
	if err := h.Load(path); err != nil {
		t.Fatal(err)
	}
	if res := h.Search(core.Vector{Embeddings: []float32{0, 1}}, 1); len(res) != 1 || res[0].Id != "b" {
		t.Errorf("expected 'b', got %v", res)
	}
}

func TestCandidateHeaps(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	var mn candMinHeap
	var mx candMaxHeap
	for i := 0; i < 500; i++ {
		d := r.Float32()
		mn.push(candidate{int32(i), d})
		mx.push(candidate{int32(i), d})
	}
	prevMin, prevMax := float32(-1), float32(2)
	for len(mn) > 0 {
		c := mn.pop()
		if c.dist < prevMin {
			t.Fatalf("min heap out of order: %v after %v", c.dist, prevMin)
		}
		prevMin = c.dist
	}
	for len(mx) > 0 {
		c := mx.pop()
		if c.dist > prevMax {
			t.Fatalf("max heap out of order: %v after %v", c.dist, prevMax)
		}
		prevMax = c.dist
	}
}

// TestHNSWConcurrentSearch checks that concurrent searches, which share the
// scratch pool, return the same results as sequential searches.
func TestHNSWConcurrentSearch(t *testing.T) {
	const n, dim = 2000, 32
	r := rand.New(rand.NewSource(7))
	idx := NewHNSWIndex(dim, 16, 100, 64, core.Euclidean)
	queries := make([]core.Vector, 32)
	for i := 0; i < n; i++ {
		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = r.Float32()*2 - 1
		}
		idx.AddVector(fmt.Sprintf("v%d", i), core.Vector{Embeddings: emb})
		if i < len(queries) {
			queries[i] = core.Vector{Embeddings: emb}
		}
	}

	want := make([][]core.SearchResult, len(queries))
	for i, q := range queries {
		want[i] = idx.Search(q, 10)
	}

	var wg sync.WaitGroup
	errs := make(chan string, 8)
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			for it := 0; it < 50; it++ {
				qi := (g + it) % len(queries)
				got := idx.Search(queries[qi], 10)
				if len(got) != len(want[qi]) {
					errs <- fmt.Sprintf("query %d: got %d results, want %d", qi, len(got), len(want[qi]))
					return
				}
				for k := range got {
					if got[k].Id != want[qi][k].Id {
						errs <- fmt.Sprintf("query %d: result %d = %s, want %s", qi, k, got[k].Id, want[qi][k].Id)
						return
					}
				}
			}
		}(g)
	}
	wg.Wait()
	close(errs)
	for e := range errs {
		t.Error(e)
	}
}

// TestHNSWConcurrentAddSearch runs inserts and searches together; run it
// with -race.
func TestHNSWConcurrentAddSearch(t *testing.T) {
	const dim = 16
	idx := NewHNSWIndex(dim, 8, 50, 32, core.Euclidean)
	vec := func(r *rand.Rand) core.Vector {
		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = r.Float32()
		}
		return core.Vector{Embeddings: emb}
	}
	seed := rand.New(rand.NewSource(3))
	for i := 0; i < 200; i++ {
		idx.AddVector(fmt.Sprintf("seed%d", i), vec(seed))
	}

	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(2)
		go func(g int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(100 + g)))
			for i := 0; i < 200; i++ {
				idx.AddVector(fmt.Sprintf("w%d-%d", g, i), vec(r))
			}
		}(g)
		go func(g int) {
			defer wg.Done()
			r := rand.New(rand.NewSource(int64(200 + g)))
			for i := 0; i < 200; i++ {
				if res := idx.Search(vec(r), 5); len(res) != 5 {
					t.Errorf("expected 5 results, got %d", len(res))
					return
				}
			}
		}(g)
	}
	wg.Wait()
}
