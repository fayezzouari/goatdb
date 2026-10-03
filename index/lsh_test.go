package index

import (
	"testing"

	"github.com/fayezzouari/goatdb/core"
)

func TestLSHIndexSearch(t *testing.T) {
	idx := NewLSHIndex(3, 10, 8, core.Euclidean)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0, 0}})
	idx.AddVector("b", core.Vector{Embeddings: []float32{0, 1, 0}})
	idx.AddVector("c", core.Vector{Embeddings: []float32{0, 0, 1}})

	results := idx.Search(core.Vector{Embeddings: []float32{1, 0, 0}}, 3)
	found := false
	for _, r := range results {
		if r.Id == "a" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("expected 'a' in results, got %v", results)
	}
}

func TestLSHIndexGetVector(t *testing.T) {
	idx := NewLSHIndex(2, 5, 4, core.Euclidean)
	idx.AddVector("x", core.Vector{Embeddings: []float32{1, 2}})

	v, ok := idx.GetVector("x")
	if !ok {
		t.Fatal("expected vector 'x' to exist")
	}
	if v.Embeddings[0] != 1 || v.Embeddings[1] != 2 {
		t.Errorf("wrong embeddings: %v", v.Embeddings)
	}
}

func TestLSHIndexDelete(t *testing.T) {
	idx := NewLSHIndex(2, 5, 4, core.Euclidean)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})
	ok := idx.DeleteVector("a")
	if !ok {
		t.Fatal("expected delete to return true")
	}
	if _, ok := idx.GetVector("a"); ok {
		t.Error("expected 'a' to be deleted")
	}
	if idx.DeleteVector("a") {
		t.Error("expected second delete to return false")
	}
}

func TestLSHIndexSaveLoad(t *testing.T) {
	path := t.TempDir() + "/lsh.bin"

	idx := NewLSHIndex(2, 5, 4, core.Cosine)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})
	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}

	idx2 := NewLSHIndex(2, 5, 4, core.Cosine)
	if err := idx2.Load(path); err != nil {
		t.Fatal(err)
	}
	if _, ok := idx2.GetVector("a"); !ok {
		t.Error("vector 'a' not found after load")
	}
}
