package index

import (
	"fmt"
	"math/rand"
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

func TestHNSWIndexDotProductOrder(t *testing.T) {
	idx := NewHNSWIndex(2, 4, 20, 10, core.DotProduct)
	idx.AddVector("low", core.Vector{Embeddings: []float32{1, 0}})
	idx.AddVector("high", core.Vector{Embeddings: []float32{5, 0}})
	idx.AddVector("mid", core.Vector{Embeddings: []float32{3, 0}})
	idx.AddVector("neg", core.Vector{Embeddings: []float32{-2, 0}})

	results := idx.Search(core.Vector{Embeddings: []float32{1, 0}}, 3)
	want := []string{"high", "mid", "low"}
	if len(results) != len(want) {
		t.Fatalf("expected %d results, got %d", len(want), len(results))
	}
	for i, id := range want {
		if results[i].Id != id {
			t.Errorf("result %d: expected %q, got %q", i, id, results[i].Id)
		}
	}
}
