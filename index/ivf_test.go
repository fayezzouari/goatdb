package index

import (
	"testing"

	"github.com/fayezzouari/goatdb/core"
)

func TestIVFIndexUntrainedFallback(t *testing.T) {
	idx := NewIVFIndex(2, 3, 2, core.Euclidean)
	// Adding without training should not panic — falls back to cluster 0
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})
	if _, ok := idx.GetVector("a"); !ok {
		t.Error("expected 'a' to be found in untrained index")
	}
}

func TestIVFIndexTrainAndSearch(t *testing.T) {
	idx := NewIVFIndex(2, 2, 2, core.Euclidean)

	vectors := []core.Vector{
		{Embeddings: []float32{1, 0}},
		{Embeddings: []float32{2, 0}},
		{Embeddings: []float32{0, 1}},
		{Embeddings: []float32{0, 2}},
	}
	idx.Train(vectors)

	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})
	idx.AddVector("b", core.Vector{Embeddings: []float32{0, 1}})

	results := idx.Search(core.Vector{Embeddings: []float32{1, 0}}, 2)
	if len(results) == 0 {
		t.Fatal("expected at least one result")
	}
	found := false
	for _, r := range results {
		if r.Id == "a" {
			found = true
		}
	}
	if !found {
		t.Errorf("expected 'a' in results, got %v", results)
	}
}

func TestIVFIndexDelete(t *testing.T) {
	idx := NewIVFIndex(2, 2, 2, core.Euclidean)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})

	if ok := idx.DeleteVector("a"); !ok {
		t.Fatal("expected delete to return true")
	}
	if _, ok := idx.GetVector("a"); ok {
		t.Error("expected 'a' to be deleted")
	}
	if idx.DeleteVector("a") {
		t.Error("expected second delete to return false")
	}
}

func TestIVFIndexSaveLoad(t *testing.T) {
	path := t.TempDir() + "/ivf.bin"

	idx := NewIVFIndex(2, 2, 2, core.Euclidean)
	vectors := []core.Vector{
		{Embeddings: []float32{1, 0}},
		{Embeddings: []float32{0, 1}},
	}
	idx.Train(vectors)
	idx.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})

	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}

	idx2 := NewIVFIndex(2, 2, 2, core.Euclidean)
	if err := idx2.Load(path); err != nil {
		t.Fatal(err)
	}
	if _, ok := idx2.GetVector("a"); !ok {
		t.Error("vector 'a' not found after load")
	}
}
