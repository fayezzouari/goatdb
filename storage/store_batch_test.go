package storage

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"
)

func TestStoreAddBatch(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}

	batch := []StoredVector{
		{Id: "a", Embeddings: []float32{1, 0}, Metadata: map[string]any{"n": "a"}},
		{Id: "b", Embeddings: []float32{0, 1}},
		{Id: "c", Embeddings: []float32{1, 1}, Metadata: map[string]any{"n": "c"}},
	}
	if err := s.AddBatch(batch); err != nil {
		t.Fatal(err)
	}
	s.Close()

	s, err = Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	seen := map[uint32]bool{}
	for _, want := range batch {
		emb, meta, err := s.Get(want.Id)
		if err != nil {
			t.Fatal(err)
		}
		if emb[0] != want.Embeddings[0] || emb[1] != want.Embeddings[1] {
			t.Errorf("%s: got %v, want %v", want.Id, emb, want.Embeddings)
		}
		if want.Metadata != nil && meta["n"] != want.Metadata["n"] {
			t.Errorf("%s: metadata got %v", want.Id, meta)
		}
		slot, _ := s.meta.GetSlot(want.Id)
		if seen[slot] {
			t.Errorf("slot %d assigned twice", slot)
		}
		seen[slot] = true
	}
}

func TestStoreAddBatchDimensionMismatch(t *testing.T) {
	s, err := Open(t.TempDir(), 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	err = s.AddBatch([]StoredVector{
		{Id: "a", Embeddings: []float32{1, 0}},
		{Id: "b", Embeddings: []float32{1, 0, 0}},
	})
	if err == nil {
		t.Fatal("expected dimension error")
	}
	if _, _, err := s.Get("a"); err == nil {
		t.Error("expected no vector written on failed batch")
	}
	if s.wal.Size() != 0 {
		t.Errorf("expected empty WAL, got %d bytes", s.wal.Size())
	}
}

func TestStoreCloseTruncatesWAL(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	s.Add("a", []float32{1, 0}, nil)
	if s.wal.Size() == 0 {
		t.Fatal("expected WAL to grow after Add")
	}
	if err := s.Close(); err != nil {
		t.Fatal(err)
	}
	info, err := os.Stat(filepath.Join(dir, "wal.log"))
	if err != nil {
		t.Fatal(err)
	}
	if info.Size() != 0 {
		t.Errorf("expected empty WAL after close, got %d bytes", info.Size())
	}
}

func TestStoreCheckpoint(t *testing.T) {
	s, err := Open(t.TempDir(), 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	s.Add("a", []float32{1, 0}, nil)
	if err := s.Checkpoint(); err != nil {
		t.Fatal(err)
	}
	if s.wal.Size() != 0 {
		t.Errorf("expected empty WAL after checkpoint, got %d bytes", s.wal.Size())
	}
	if emb, _, err := s.Get("a"); err != nil || emb[0] != 1 {
		t.Errorf("unexpected state after checkpoint: %v %v", emb, err)
	}
}

// crash closes the store's files without flushing the vector file or
// truncating the WAL.
func crash(t *testing.T, s *Store) {
	t.Helper()
	s.wal.Close()
	syscall.Munmap(s.vectors.data)
	s.vectors.f.Close()
	s.meta.Close()
}

// clobberSlot overwrites a record in vectors.bin to simulate a write that
// never reached disk.
func clobberSlot(t *testing.T, dir string, dim int, slot uint32, deleted bool) {
	t.Helper()
	f, err := os.OpenFile(filepath.Join(dir, "vectors.bin"), os.O_RDWR, 0644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	rec := make([]byte, dim*4+1)
	if deleted {
		rec[dim*4] = deletedFlag
	}
	if _, err := f.WriteAt(rec, int64(slot)*int64(len(rec))); err != nil {
		t.Fatal(err)
	}
}

func TestStoreReplayReappliesCommittedInsert(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	s.AddBatch([]StoredVector{
		{Id: "a", Embeddings: []float32{1, 2}},
		{Id: "b", Embeddings: []float32{3, 4}},
	})
	slotA, _ := s.meta.GetSlot("a")
	slotB, _ := s.meta.GetSlot("b")
	crash(t, s)
	clobberSlot(t, dir, 2, slotA, false)
	clobberSlot(t, dir, 2, slotB, true)

	s, err = Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	for id, want := range map[string][]float32{"a": {1, 2}, "b": {3, 4}} {
		emb, _, err := s.Get(id)
		if err != nil {
			t.Fatalf("%s: %v", id, err)
		}
		if emb[0] != want[0] || emb[1] != want[1] {
			t.Errorf("%s: got %v, want %v", id, emb, want)
		}
	}
}

func TestStoreReplaySkipsUncommittedInsert(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	s.Add("a", []float32{1, 2}, nil)
	slot, _ := s.meta.GetSlot("a")
	// WAL record whose metadata transaction never committed
	s.wal.Append(opInsert, "ghost", slot, []float32{9, 9})
	s.wal.Sync()
	crash(t, s)

	s, err = Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	if _, _, err := s.Get("ghost"); err == nil {
		t.Error("uncommitted insert should not be visible")
	}
	emb, _, err := s.Get("a")
	if err != nil {
		t.Fatal(err)
	}
	if emb[0] != 1 || emb[1] != 2 {
		t.Errorf("uncommitted insert overwrote slot: got %v", emb)
	}
}

func TestStoreReplayDeleteAndReinsert(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	s.Add("a", []float32{1, 0}, nil)
	s.Add("b", []float32{5, 5}, nil)
	slotB, _ := s.meta.GetSlot("b")
	slotOld, _ := s.meta.GetSlot("a")
	s.Delete("a")
	s.Delete("b")
	s.Add("a", []float32{0, 1}, map[string]any{"v": "2"})
	slotA, _ := s.meta.GetSlot("a")
	if slotA != slotOld || slotA == slotB {
		t.Fatalf("expected a to reuse its old slot %d, got %d (b=%d)", slotOld, slotA, slotB)
	}
	crash(t, s)
	clobberSlot(t, dir, 2, slotA, false)
	clobberSlot(t, dir, 2, slotB, false)

	s, err = Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	emb, meta, err := s.Get("a")
	if err != nil {
		t.Fatal(err)
	}
	if emb[0] != 0 || emb[1] != 1 || meta["v"] != "2" {
		t.Errorf("got %v %v, want [0 1] map[v:2]", emb, meta)
	}
	if _, _, err := s.Get("b"); err == nil {
		t.Error("expected b to stay deleted")
	}
	if _, ok := s.vectors.Read(int(slotB)); ok {
		t.Error("expected delete flag on b's slot to be re-applied")
	}
}
