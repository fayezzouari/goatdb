package storage

import (
	"testing"
)

func TestWALAppendAndReplay(t *testing.T) {
	path := t.TempDir() + "/wal.log"

	wal, _, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}

	emb := []float32{1.0, 2.0, 3.0}
	if err := wal.Append(opInsert, "v1", 0, emb); err != nil {
		t.Fatal(err)
	}
	if err := wal.Sync(); err != nil {
		t.Fatal(err)
	}
	wal.Close()

	_, entries, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}

	e := entries[0]
	if e.op != opInsert {
		t.Errorf("expected opInsert, got %v", e.op)
	}
	if e.id != "v1" {
		t.Errorf("expected id 'v1', got %q", e.id)
	}
	if e.slot != 0 {
		t.Errorf("expected slot 0, got %d", e.slot)
	}
	for i, v := range emb {
		if e.embeddings[i] != v {
			t.Errorf("embedding[%d]: got %v, want %v", i, e.embeddings[i], v)
		}
	}
}

func TestWALMultipleOps(t *testing.T) {
	path := t.TempDir() + "/wal.log"

	wal, _, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}

	wal.Append(opInsert, "a", 0, []float32{1, 0})
	wal.Append(opUpdate, "a", 0, []float32{0, 1})
	wal.Append(opDelete, "a", 0, nil)
	wal.Sync()
	wal.Close()

	_, entries, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 3 {
		t.Fatalf("expected 3 entries, got %d", len(entries))
	}
	if entries[0].op != opInsert || entries[1].op != opUpdate || entries[2].op != opDelete {
		t.Errorf("wrong ops: %v %v %v", entries[0].op, entries[1].op, entries[2].op)
	}
}

func TestWALTruncate(t *testing.T) {
	path := t.TempDir() + "/wal.log"

	wal, _, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	wal.Append(opInsert, "v1", 0, []float32{1, 2})
	wal.Sync()
	if err := wal.Truncate(); err != nil {
		t.Fatal(err)
	}
	wal.Close()

	_, entries, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Errorf("expected 0 entries after truncate, got %d", len(entries))
	}
}

func TestWALDeleteHasNoEmbeddings(t *testing.T) {
	path := t.TempDir() + "/wal.log"

	wal, _, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	wal.Append(opDelete, "v1", 2, nil)
	wal.Sync()
	wal.Close()

	_, entries, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 {
		t.Fatalf("expected 1 entry, got %d", len(entries))
	}
	if len(entries[0].embeddings) != 0 {
		t.Errorf("delete entry should have no embeddings, got %v", entries[0].embeddings)
	}
}
