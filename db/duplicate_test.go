package db

import (
	"errors"
	"testing"

	"github.com/fayez/goatdb/core"
)

func TestCollectionAddExistingID(t *testing.T) {
	col := newTestCollection(t)

	if err := col.AddVector(ctx, "v1", core.Vector{Embeddings: []float32{1, 0}, Metadata: map[string]any{"n": 1}}); err != nil {
		t.Fatal(err)
	}
	err := col.AddVector(ctx, "v1", core.Vector{Embeddings: []float32{0, 1}})
	if !errors.Is(err, ErrExists) {
		t.Fatalf("expected ErrExists, got %v", err)
	}

	got, ok := col.GetVector(ctx, "v1")
	if !ok || got.Embeddings[0] != 1 || got.Embeddings[1] != 0 {
		t.Errorf("original vector changed: %v", got)
	}
	results, err := col.Search(ctx, core.Vector{Embeddings: []float32{1, 0}}, 10)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 1 {
		t.Errorf("expected a single index entry, got %d", len(results))
	}
}

func TestCollectionAddVectors(t *testing.T) {
	col := newTestCollection(t)

	err := col.AddVectors(ctx, []VectorEntry{
		{Id: "a", Vector: core.Vector{Embeddings: []float32{1, 0}}},
		{Id: "b", Vector: core.Vector{Embeddings: []float32{0, 1}, Metadata: map[string]any{"tag": "b"}}},
	})
	if err != nil {
		t.Fatal(err)
	}
	got, ok := col.GetVector(ctx, "b")
	if !ok || got.Metadata["tag"] != "b" {
		t.Errorf("unexpected b: %v %v", got, ok)
	}
	results, _ := col.Search(ctx, core.Vector{Embeddings: []float32{1, 0}}, 10)
	if len(results) != 2 {
		t.Errorf("expected 2 indexed vectors, got %d", len(results))
	}
}

func TestCollectionAddVectorsExistingIDRejectsBatch(t *testing.T) {
	col := newTestCollection(t)

	col.AddVector(ctx, "a", core.Vector{Embeddings: []float32{1, 0}})
	err := col.AddVectors(ctx, []VectorEntry{
		{Id: "b", Vector: core.Vector{Embeddings: []float32{0, 1}}},
		{Id: "a", Vector: core.Vector{Embeddings: []float32{1, 1}}},
	})
	if !errors.Is(err, ErrExists) {
		t.Fatalf("expected ErrExists, got %v", err)
	}
	if _, ok := col.GetVector(ctx, "b"); ok {
		t.Error("expected no vector from a rejected batch to be written")
	}
	results, _ := col.Search(ctx, core.Vector{Embeddings: []float32{1, 0}}, 10)
	if len(results) != 1 {
		t.Errorf("expected index unchanged, got %d entries", len(results))
	}
}

func TestCollectionAddVectorsDuplicateInBatch(t *testing.T) {
	col := newTestCollection(t)

	err := col.AddVectors(ctx, []VectorEntry{
		{Id: "x", Vector: core.Vector{Embeddings: []float32{1, 0}}},
		{Id: "y", Vector: core.Vector{Embeddings: []float32{0, 1}}},
		{Id: "x", Vector: core.Vector{Embeddings: []float32{1, 1}}},
	})
	if !errors.Is(err, ErrDuplicateID) {
		t.Fatalf("expected ErrDuplicateID, got %v", err)
	}
	for _, id := range []string{"x", "y"} {
		if _, ok := col.GetVector(ctx, id); ok {
			t.Errorf("expected %q not to be written", id)
		}
	}
}

func TestCollectionAddVectorsDimensionMismatchWritesNothing(t *testing.T) {
	col := newTestCollection(t)

	err := col.AddVectors(ctx, []VectorEntry{
		{Id: "ok", Vector: core.Vector{Embeddings: []float32{1, 0}}},
		{Id: "bad", Vector: core.Vector{Embeddings: []float32{1, 0, 0}}},
	})
	if err == nil {
		t.Fatal("expected dimension mismatch error")
	}
	if _, ok := col.GetVector(ctx, "ok"); ok {
		t.Error("expected no vector from a rejected batch to be written")
	}
}
