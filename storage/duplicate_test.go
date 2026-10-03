package storage

import (
	"errors"
	"testing"
)

func TestStoreAddExistingID(t *testing.T) {
	s, err := Open(t.TempDir(), 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if err := s.Add("a", []float32{1, 0}, map[string]any{"n": "1"}); err != nil {
		t.Fatal(err)
	}
	walSize := s.wal.Size()

	if err := s.Add("a", []float32{0, 1}, nil); !errors.Is(err, ErrExists) {
		t.Fatalf("expected ErrExists, got %v", err)
	}
	if s.wal.Size() != walSize {
		t.Error("rejected insert should not be written to the WAL")
	}
	emb, meta, err := s.Get("a")
	if err != nil {
		t.Fatal(err)
	}
	if emb[0] != 1 || meta["n"] != "1" {
		t.Errorf("original vector changed: %v %v", emb, meta)
	}

	// the rejected insert must not have consumed a slot
	if err := s.Add("b", []float32{0, 1}, nil); err != nil {
		t.Fatal(err)
	}
	if slot, _ := s.meta.GetSlot("b"); slot != 1 {
		t.Errorf("expected b in slot 1, got %d", slot)
	}
}

func TestStoreAddBatchRejectsWholeBatch(t *testing.T) {
	s, err := Open(t.TempDir(), 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	s.Add("a", []float32{1, 0}, nil)

	err = s.AddBatch([]StoredVector{
		{Id: "b", Embeddings: []float32{0, 1}},
		{Id: "a", Embeddings: []float32{1, 1}},
	})
	if !errors.Is(err, ErrExists) {
		t.Fatalf("expected ErrExists, got %v", err)
	}
	if _, _, err := s.Get("b"); err == nil {
		t.Error("expected b not to be written")
	}

	err = s.AddBatch([]StoredVector{
		{Id: "c", Embeddings: []float32{0, 1}},
		{Id: "c", Embeddings: []float32{1, 1}},
	})
	if !errors.Is(err, ErrDuplicateID) {
		t.Fatalf("expected ErrDuplicateID, got %v", err)
	}
	if _, _, err := s.Get("c"); err == nil {
		t.Error("expected c not to be written")
	}
}
