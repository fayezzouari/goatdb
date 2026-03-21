package db

import (
	"testing"

	"github.com/fayez/goatdb/core"
	"github.com/fayez/goatdb/index"
)

func newTestCollection(t *testing.T) *Collection {
	t.Helper()
	dir := t.TempDir()
	idx := index.NewFlatIndex(2, core.Euclidean)
	col, err := newCollection("test", 2, dir, idx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { col.Close() })
	return col
}

func TestCollectionAddGet(t *testing.T) {
	col := newTestCollection(t)

	v := core.Vector{Embeddings: []float32{1, 0}, Metadata: map[string]any{"tag": "a"}}
	if err := col.AddVector("v1", v); err != nil {
		t.Fatal(err)
	}

	got, ok := col.GetVector("v1")
	if !ok {
		t.Fatal("vector not found")
	}
	if got.Embeddings[0] != 1 || got.Embeddings[1] != 0 {
		t.Errorf("wrong embeddings: %v", got.Embeddings)
	}
	if got.Metadata["tag"] != "a" {
		t.Errorf("wrong metadata: %v", got.Metadata)
	}
}

func TestCollectionDimensionMismatch(t *testing.T) {
	col := newTestCollection(t)
	err := col.AddVector("v1", core.Vector{Embeddings: []float32{1, 2, 3}})
	if err == nil {
		t.Error("expected dimension mismatch error")
	}
}

func TestCollectionUpdate(t *testing.T) {
	col := newTestCollection(t)

	col.AddVector("v1", core.Vector{Embeddings: []float32{1, 0}})
	if err := col.UpdateVector("v1", core.Vector{Embeddings: []float32{0, 1}, Metadata: map[string]any{"updated": true}}); err != nil {
		t.Fatal(err)
	}

	got, ok := col.GetVector("v1")
	if !ok {
		t.Fatal("vector not found after update")
	}
	if got.Embeddings[0] != 0 || got.Embeddings[1] != 1 {
		t.Errorf("wrong embeddings after update: %v", got.Embeddings)
	}
}

func TestCollectionDelete(t *testing.T) {
	col := newTestCollection(t)

	col.AddVector("v1", core.Vector{Embeddings: []float32{1, 0}})
	if err := col.DeleteVector("v1"); err != nil {
		t.Fatal(err)
	}

	if _, ok := col.GetVector("v1"); ok {
		t.Error("expected vector to be deleted")
	}
}

func TestCollectionSearch(t *testing.T) {
	col := newTestCollection(t)

	col.AddVector("a", core.Vector{Embeddings: []float32{1, 0}})
	col.AddVector("b", core.Vector{Embeddings: []float32{0, 1}})

	results, err := col.Search(core.Vector{Embeddings: []float32{1, 0}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Id != "a" {
		t.Errorf("expected 'a', got %v", results)
	}
}

func TestCollectionPersistence(t *testing.T) {
	dir := t.TempDir()

	db, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}

	col, err := db.CreateCollection("vecs", 2, core.Euclidean, "flat")
	if err != nil {
		t.Fatal(err)
	}
	col.AddVector("x", core.Vector{Embeddings: []float32{1, 1}})
	db.Close()

	db2, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()

	col2, err := db2.GetCollection("vecs")
	if err != nil {
		t.Fatal(err)
	}

	results, err := col2.Search(core.Vector{Embeddings: []float32{1, 1}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 || results[0].Id != "x" {
		t.Errorf("expected 'x' after reload, got %v", results)
	}
}
