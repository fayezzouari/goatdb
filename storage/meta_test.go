package storage

import (
	"encoding/binary"
	"errors"
	"fmt"
	"testing"

	bolt "go.etcd.io/bbolt"
)

func TestMetaStoreAllocSlot(t *testing.T) {
	path := t.TempDir() + "/meta.db"
	ms, err := openMetaStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ms.Close()

	slot0, err := ms.AllocSlot("a")
	if err != nil {
		t.Fatal(err)
	}
	slot1, err := ms.AllocSlot("b")
	if err != nil {
		t.Fatal(err)
	}
	if slot0 == slot1 {
		t.Errorf("expected distinct slots, both got %d", slot0)
	}
}

func TestMetaStoreGetSlot(t *testing.T) {
	path := t.TempDir() + "/meta.db"
	ms, err := openMetaStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ms.Close()

	slot, _ := ms.AllocSlot("x")
	got, err := ms.GetSlot("x")
	if err != nil {
		t.Fatal(err)
	}
	if got != slot {
		t.Errorf("GetSlot: got %d, want %d", got, slot)
	}

	_, err = ms.GetSlot("missing")
	if !errors.Is(err, ErrNotFound) {
		t.Errorf("expected ErrNotFound, got %v", err)
	}
}

func TestMetaStorePutGetMeta(t *testing.T) {
	path := t.TempDir() + "/meta.db"
	ms, err := openMetaStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ms.Close()

	meta := map[string]any{"label": "test", "score": 1.0}
	if err := ms.PutMeta("v1", meta); err != nil {
		t.Fatal(err)
	}

	got, err := ms.GetMeta("v1")
	if err != nil {
		t.Fatal(err)
	}
	if got["label"] != "test" {
		t.Errorf("expected label=test, got %v", got["label"])
	}
}

func TestMetaStoreDelete(t *testing.T) {
	path := t.TempDir() + "/meta.db"
	ms, err := openMetaStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ms.Close()

	ms.AllocSlot("a")
	ms.PutMeta("a", map[string]any{"x": 1})

	if err := ms.Delete("a"); err != nil {
		t.Fatal(err)
	}

	if _, err := ms.GetSlot("a"); !errors.Is(err, ErrNotFound) {
		t.Error("expected slot to be gone after delete")
	}
}

func TestMetaStoreSlotReuse(t *testing.T) {
	path := t.TempDir() + "/meta.db"
	ms, err := openMetaStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ms.Close()

	slot, _ := ms.AllocSlot("a")
	ms.Delete("a")

	// Next alloc should reuse the freed slot
	reused, err := ms.AllocSlot("b")
	if err != nil {
		t.Fatal(err)
	}
	if reused != slot {
		t.Errorf("expected slot reuse: got %d, want %d", reused, slot)
	}
}

func TestMetaStoreAllSlots(t *testing.T) {
	path := t.TempDir() + "/meta.db"
	ms, err := openMetaStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ms.Close()

	ms.AllocSlot("a")
	ms.AllocSlot("b")
	ms.AllocSlot("c")
	ms.Delete("b")

	ids, slots, err := ms.AllSlots()
	if err != nil {
		t.Fatal(err)
	}
	if len(ids) != 2 || len(slots) != 2 {
		t.Errorf("expected 2 slots after delete, got %d", len(ids))
	}
	for _, id := range ids {
		if id == "b" {
			t.Error("deleted id 'b' should not appear in AllSlots")
		}
	}
}

func TestMetaStoreFreeSlotsManyDeletes(t *testing.T) {
	path := t.TempDir() + "/meta.db"
	ms, err := openMetaStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ms.Close()

	const n = 500
	for i := 0; i < n; i++ {
		if _, err := ms.AllocSlot(fmt.Sprintf("v%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	for i := 0; i < n; i += 2 {
		if err := ms.Delete(fmt.Sprintf("v%d", i)); err != nil {
			t.Fatal(err)
		}
	}
	seen := map[uint32]bool{}
	for i := 0; i < n/2; i++ {
		slot, err := ms.AllocSlot(fmt.Sprintf("r%d", i))
		if err != nil {
			t.Fatal(err)
		}
		if slot%2 != 0 || seen[slot] {
			t.Fatalf("alloc %d: got slot %d, want an unused even slot", i, slot)
		}
		seen[slot] = true
	}
	if slot, _ := ms.AllocSlot("fresh"); slot != n {
		t.Errorf("expected fresh slot %d once free list is empty, got %d", n, slot)
	}
}

func TestMetaStoreMigratesLegacyFreeSlots(t *testing.T) {
	path := t.TempDir() + "/meta.db"
	ms, err := openMetaStore(path)
	if err != nil {
		t.Fatal(err)
	}
	legacy := make([]byte, 8)
	binary.LittleEndian.PutUint32(legacy[0:], 7)
	binary.LittleEndian.PutUint32(legacy[4:], 3)
	err = ms.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(bucketConfig).Put(keyFreeSlots, legacy)
	})
	if err != nil {
		t.Fatal(err)
	}
	ms.Close()

	ms, err = openMetaStore(path)
	if err != nil {
		t.Fatal(err)
	}
	defer ms.Close()
	err = ms.db.View(func(tx *bolt.Tx) error {
		if tx.Bucket(bucketConfig).Get(keyFreeSlots) != nil {
			t.Error("legacy free list was not removed")
		}
		return nil
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []uint32{3, 7} {
		got, err := ms.AllocSlot(fmt.Sprintf("m%d", want))
		if err != nil {
			t.Fatal(err)
		}
		if got != want {
			t.Errorf("expected migrated slot %d, got %d", want, got)
		}
	}
}
