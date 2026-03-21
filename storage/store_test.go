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
