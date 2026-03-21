package storage

import (
	"testing"
)

func TestVectorStoreWriteRead(t *testing.T) {
	path := t.TempDir() + "/vectors.bin"
	vs, err := openVectorStore(path, 3)
	if err != nil {
		t.Fatal(err)
	}
	defer vs.Close()

	emb := []float32{1.5, 2.5, 3.5}
	if err := vs.Write(0, emb); err != nil {
		t.Fatal(err)
	}

	got, ok := vs.Read(0)
	if !ok {
		t.Fatal("expected slot 0 to be readable")
	}
	for i, v := range emb {
		if got[i] != v {
			t.Errorf("slot 0 dim %d: got %v, want %v", i, got[i], v)
		}
	}
}

func TestVectorStoreDelete(t *testing.T) {
	path := t.TempDir() + "/vectors.bin"
	vs, err := openVectorStore(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer vs.Close()

	vs.Write(0, []float32{1, 2})
	vs.Delete(0)

	if _, ok := vs.Read(0); ok {
		t.Error("expected slot 0 to be deleted")
	}
}

func TestVectorStoreMultipleSlots(t *testing.T) {
	path := t.TempDir() + "/vectors.bin"
	vs, err := openVectorStore(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer vs.Close()

	vs.Write(0, []float32{1, 0})
	vs.Write(1, []float32{0, 1})
	vs.Write(2, []float32{2, 2})

	for slot, want := range [][]float32{{1, 0}, {0, 1}, {2, 2}} {
		got, ok := vs.Read(slot)
		if !ok {
			t.Fatalf("slot %d not readable", slot)
		}
		for i, v := range want {
			if got[i] != v {
				t.Errorf("slot %d dim %d: got %v, want %v", slot, i, got[i], v)
			}
		}
	}
}

func TestVectorStoreGrow(t *testing.T) {
	path := t.TempDir() + "/vectors.bin"
	vs, err := openVectorStore(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer vs.Close()

	// Write beyond initial capacity (1024 slots) to trigger grow
	slot := 2000
	emb := []float32{7, 8}
	if err := vs.Write(slot, emb); err != nil {
		t.Fatal(err)
	}
	got, ok := vs.Read(slot)
	if !ok {
		t.Fatal("expected slot to be readable after grow")
	}
	if got[0] != 7 || got[1] != 8 {
		t.Errorf("wrong values after grow: %v", got)
	}
}

func TestVectorStoreReadOutOfBounds(t *testing.T) {
	path := t.TempDir() + "/vectors.bin"
	vs, err := openVectorStore(path, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer vs.Close()

	if _, ok := vs.Read(99999); ok {
		t.Error("expected out-of-bounds read to return false")
	}
}
