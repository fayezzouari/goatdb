package index

import (
	"testing"

	"github.com/fayez/goatdb/core"
)

func TestFlatIndexSearch(t *testing.T) {
	idx := NewFlatIndex(3, core.Euclidean)
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

func TestFlatIndexTopK(t *testing.T) {
	idx := NewFlatIndex(2, core.Euclidean)
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

func TestFlatIndexDelete(t *testing.T) {
	idx := NewFlatIndex(2, core.Euclidean)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})
	idx.DeleteVector("a")

	if _, ok := idx.GetVector("a"); ok {
		t.Error("expected vector to be deleted")
	}
	results := idx.Search(core.Vector{Embeddings: []float32{1, 0}}, 10)
	if len(results) != 0 {
		t.Errorf("expected no results after delete, got %d", len(results))
	}
}

func TestFlatIndexSaveLoad(t *testing.T) {
	path := t.TempDir() + "/flat.bin"

	idx := NewFlatIndex(2, core.Cosine)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})
	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}

	idx2 := NewFlatIndex(2, core.Cosine)
	if err := idx2.Load(path); err != nil {
		t.Fatal(err)
	}
	if _, ok := idx2.GetVector("a"); !ok {
		t.Error("vector 'a' not found after load")
	}
}
