package db

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/fayez/goatdb/core"
	"github.com/fayez/goatdb/index"
)

var ctx = context.Background()

func newTestCollection(t *testing.T) *Collection {
	t.Helper()
	dir := t.TempDir()
	idx := index.NewFlatIndex(2, core.Euclidean)
	col, err := newCollection("test", 2, core.Euclidean, "flat", dir, idx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { col.Close() })
	return col
}

func TestCollectionAddGet(t *testing.T) {
	col := newTestCollection(t)

	v := core.Vector{Embeddings: []float32{1, 0}, Metadata: map[string]any{"tag": "a"}}
	if err := col.AddVector(ctx, "v1", v); err != nil {
		t.Fatal(err)
	}

	got, ok := col.GetVector(ctx, "v1")
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
	err := col.AddVector(ctx, "v1", core.Vector{Embeddings: []float32{1, 2, 3}})
	if err == nil {
		t.Error("expected dimension mismatch error")
	}
}

func TestCollectionUpdate(t *testing.T) {
	col := newTestCollection(t)

	col.AddVector(ctx, "v1", core.Vector{Embeddings: []float32{1, 0}})
	if err := col.UpdateVector(ctx, "v1", core.Vector{Embeddings: []float32{0, 1}, Metadata: map[string]any{"updated": true}}); err != nil {
		t.Fatal(err)
	}

	got, ok := col.GetVector(ctx, "v1")
	if !ok {
		t.Fatal("vector not found after update")
	}
	if got.Embeddings[0] != 0 || got.Embeddings[1] != 1 {
		t.Errorf("wrong embeddings after update: %v", got.Embeddings)
	}
}

func TestCollectionDelete(t *testing.T) {
	col := newTestCollection(t)

	col.AddVector(ctx, "v1", core.Vector{Embeddings: []float32{1, 0}})
	if err := col.DeleteVector(ctx, "v1"); err != nil {
		t.Fatal(err)
	}

	if _, ok := col.GetVector(ctx, "v1"); ok {
		t.Error("expected vector to be deleted")
	}
}

func TestCollectionSearch(t *testing.T) {
	col := newTestCollection(t)

	col.AddVector(ctx, "a", core.Vector{Embeddings: []float32{1, 0}})
	col.AddVector(ctx, "b", core.Vector{Embeddings: []float32{0, 1}})

	results, err := col.Search(ctx, core.Vector{Embeddings: []float32{1, 0}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 || results[0].Id != "a" {
		t.Errorf("expected 'a', got %v", results)
	}
}

func TestCollectionPersistence(t *testing.T) {
	dir := t.TempDir()

	database, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}

	col, err := database.CreateCollection("vecs", 2, core.Euclidean, "flat")
	if err != nil {
		t.Fatal(err)
	}
	col.AddVector(ctx, "x", core.Vector{Embeddings: []float32{1, 1}})
	database.Close()

	db2, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer db2.Close()

	col2, err := db2.GetCollection("vecs")
	if err != nil {
		t.Fatal(err)
	}

	results, err := col2.Search(ctx, core.Vector{Embeddings: []float32{1, 1}}, 1)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) == 0 || results[0].Id != "x" {
		t.Errorf("expected 'x' after reload, got %v", results)
	}
}

// TestCollectionTrainConcurrent runs Train alongside writes and searches on
// an HNSW collection; run it with -race.
func TestCollectionTrainConcurrent(t *testing.T) {
	const dim = 8
	idx := index.NewHNSWIndex(dim, 8, 32, 16, core.Euclidean)
	col, err := newCollection("train", dim, core.Euclidean, "hnsw", t.TempDir(), idx)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { col.Close() })

	vec := func(i int) core.Vector {
		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = float32((i*7+j*13)%101) / 101
		}
		return core.Vector{Embeddings: emb}
	}
	for i := 0; i < 50; i++ {
		if err := col.AddVector(ctx, fmt.Sprintf("seed%d", i), vec(i)); err != nil {
			t.Fatal(err)
		}
	}

	var wg sync.WaitGroup
	wg.Add(3)
	go func() {
		defer wg.Done()
		for i := 0; i < 3; i++ {
			if err := col.Train(ctx); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 30; i++ {
			if err := col.AddVector(ctx, fmt.Sprintf("w%d", i), vec(1000+i)); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			if _, err := col.Search(ctx, vec(i), 5); err != nil {
				t.Error(err)
				return
			}
		}
	}()
	wg.Wait()
}

func TestCollectionSearchWithOptions(t *testing.T) {
	col := newTestCollection(t)
	col.AddVector(ctx, "a", core.Vector{Embeddings: []float32{1, 0}, Metadata: map[string]any{"tag": "a"}})
	col.AddVector(ctx, "b", core.Vector{Embeddings: []float32{0, 1}, Metadata: map[string]any{"tag": "b"}})
	q := core.Vector{Embeddings: []float32{1, 0}}

	cases := []struct {
		name     string
		opts     SearchOptions
		wantEmb  bool
		wantMeta bool
	}{
		{"none", SearchOptions{}, false, false},
		{"metadata", SearchOptions{IncludeMetadata: true}, false, true},
		{"vectors", SearchOptions{IncludeVectors: true}, true, false},
		{"both", SearchOptions{IncludeVectors: true, IncludeMetadata: true}, true, true},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			results, err := col.SearchWithOptions(ctx, q, 2, tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			if len(results) != 2 || results[0].Id != "a" {
				t.Fatalf("expected 'a' first of 2, got %v", results)
			}
			if got := results[0].Vector.Embeddings != nil; got != tc.wantEmb {
				t.Errorf("embeddings present = %v, want %v", got, tc.wantEmb)
			}
			if got := results[0].Vector.Metadata != nil; got != tc.wantMeta {
				t.Errorf("metadata present = %v, want %v", got, tc.wantMeta)
			}
			if tc.wantMeta && results[0].Vector.Metadata["tag"] != "a" {
				t.Errorf("wrong metadata: %v", results[0].Vector.Metadata)
			}
		})
	}
}
