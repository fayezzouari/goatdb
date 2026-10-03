package storage

import (
	"fmt"
	"testing"
)

func TestStoreAddGet(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	emb := []float32{1, 2}
	meta := map[string]any{"label": "test"}
	if err := s.Add("v1", emb, meta); err != nil {
		t.Fatal(err)
	}

	gotEmb, gotMeta, err := s.Get("v1")
	if err != nil {
		t.Fatal(err)
	}
	if gotEmb[0] != 1 || gotEmb[1] != 2 {
		t.Errorf("embeddings mismatch: got %v", gotEmb)
	}
	if gotMeta["label"] != "test" {
		t.Errorf("metadata mismatch: got %v", gotMeta)
	}
}

func TestStoreUpdate(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	s.Add("v1", []float32{1, 0}, map[string]any{"x": 1})
	if err := s.Update("v1", []float32{0, 1}, map[string]any{"x": 2}); err != nil {
		t.Fatal(err)
	}

	emb, meta, err := s.Get("v1")
	if err != nil {
		t.Fatal(err)
	}
	if emb[0] != 0 || emb[1] != 1 {
		t.Errorf("expected updated embeddings, got %v", emb)
	}
	if fmt.Sprintf("%v", meta["x"]) != "2" {
		t.Errorf("expected updated metadata x=2, got %v", meta)
	}
}

func TestStoreDelete(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	s.Add("v1", []float32{1, 0}, nil)
	if err := s.Delete("v1"); err != nil {
		t.Fatal(err)
	}

	if _, _, err := s.Get("v1"); err == nil {
		t.Error("expected error after delete")
	}
}

func TestStoreLoadEmbeddings(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}

	s.Add("a", []float32{1, 0}, nil)
	s.Add("b", []float32{0, 1}, nil)
	s.Delete("a")

	vecs, err := s.LoadEmbeddings()
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 1 || vecs[0].Id != "b" {
		t.Errorf("expected only 'b', got %v", vecs)
	}
	s.Close()
}

func TestStoreWALReplay(t *testing.T) {
	dir := t.TempDir()

	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	s.Add("v1", []float32{3, 4}, map[string]any{"k": "v"})
	s.Close()

	s2, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s2.Close()

	emb, _, err := s2.Get("v1")
	if err != nil {
		t.Fatal(err)
	}
	if emb[0] != 3 || emb[1] != 4 {
		t.Errorf("expected {3,4} after reload, got %v", emb)
	}
}

func TestStoreGetMany(t *testing.T) {
	s, err := Open(t.TempDir(), 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	s.Add("a", []float32{1, 2}, map[string]any{"label": "a"})
	s.Add("b", []float32{3, 4}, nil)
	s.Add("c", []float32{5, 6}, map[string]any{"label": "c"})
	s.Delete("c")

	ids := []string{"a", "missing", "b", "c"}

	got, err := s.GetMany(ids, true, true)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(ids) {
		t.Fatalf("expected %d results, got %d", len(ids), len(got))
	}
	if got[0].Id != "a" || got[0].Embeddings[1] != 2 || got[0].Metadata["label"] != "a" {
		t.Errorf("unexpected result for a: %+v", got[0])
	}
	if got[1].Id != "" {
		t.Errorf("missing id should be empty, got %+v", got[1])
	}
	if got[2].Id != "b" || got[2].Embeddings[0] != 3 || got[2].Metadata != nil {
		t.Errorf("unexpected result for b: %+v", got[2])
	}
	if got[3].Id != "" {
		t.Errorf("deleted id should be empty, got %+v", got[3])
	}

	got, err = s.GetMany(ids, false, true)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Id != "a" || got[0].Embeddings != nil || got[0].Metadata["label"] != "a" {
		t.Errorf("expected metadata only for a, got %+v", got[0])
	}
	if got[3].Id != "" {
		t.Errorf("deleted id should be empty without embeddings too, got %+v", got[3])
	}

	got, err = s.GetMany(ids, false, false)
	if err != nil {
		t.Fatal(err)
	}
	if got[0].Id != "a" || got[0].Embeddings != nil || got[0].Metadata != nil {
		t.Errorf("expected id only for a, got %+v", got[0])
	}
}
