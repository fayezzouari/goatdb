package storage

import (
	"encoding/binary"
	"errors"
	"fmt"

	msgpack "github.com/shamaton/msgpack/v2"
	bolt "go.etcd.io/bbolt"
)

var (
	bucketSlots  = []byte("slots")
	bucketMeta   = []byte("meta")
	bucketConfig = []byte("config")
	bucketFree   = []byte("free_slots")
	keyNextSlot  = []byte("next_slot")
	keyFreeSlots = []byte("free_slots")
)

var (
	ErrNotFound    = errors.New("not found")
	ErrExists      = errors.New("already exists")
	ErrDuplicateID = errors.New("duplicate id in batch")
)

type MetaStore struct {
	db *bolt.DB
}

func openMetaStore(path string) (*MetaStore, error) {
	db, err := bolt.Open(path, 0644, nil)
	if err != nil {
		return nil, err
	}
	err = db.Update(func(tx *bolt.Tx) error {
		for _, b := range [][]byte{bucketSlots, bucketMeta, bucketConfig, bucketFree} {
			if _, err := tx.CreateBucketIfNotExists(b); err != nil {
				return err
			}
		}
		return migrateFreeSlots(tx)
	})
	return &MetaStore{db: db}, err
}

func (m *MetaStore) AllocSlot(id string) (uint32, error) {
	var slot uint32
	err := m.db.Update(func(tx *bolt.Tx) error {
		slots := allocSlots(tx, 1)
		slot = slots[0]
		return putSlot(tx, id, slot)
	})
	return slot, err
}

// Insert allocates a slot for each id and stores its metadata in a single
// transaction. It fails with ErrExists if any id is already present.
// beforeCommit runs inside the transaction once slots are known; if it fails,
// nothing is committed.
func (m *MetaStore) Insert(ids []string, metas [][]byte, beforeCommit func(slots []uint32) error) ([]uint32, error) {
	var slots []uint32
	err := m.db.Update(func(tx *bolt.Tx) error {
		slotBucket := tx.Bucket(bucketSlots)
		for _, id := range ids {
			if slotBucket.Get([]byte(id)) != nil {
				return fmt.Errorf("vector %q %w", id, ErrExists)
			}
		}
		slots = allocSlots(tx, len(ids))
		metaBucket := tx.Bucket(bucketMeta)
		for i, id := range ids {
			if err := putSlot(tx, id, slots[i]); err != nil {
				return err
			}
			if err := metaBucket.Put([]byte(id), metas[i]); err != nil {
				return err
			}
		}
		return beforeCommit(slots)
	})
	if err != nil {
		return nil, err
	}
	return slots, nil
}

func allocSlots(tx *bolt.Tx, n int) []uint32 {
	cfg := tx.Bucket(bucketConfig)
	slots := make([]uint32, 0, n)

	c := tx.Bucket(bucketFree).Cursor()
	for k, _ := c.First(); k != nil && len(slots) < n; k, _ = c.First() {
		slots = append(slots, binary.BigEndian.Uint32(k))
		c.Delete()
	}

	if len(slots) < n {
		var next uint64
		if raw := cfg.Get(keyNextSlot); raw != nil {
			next = binary.LittleEndian.Uint64(raw)
		}
		for len(slots) < n {
			slots = append(slots, uint32(next))
			next++
		}
		var buf [8]byte
		binary.LittleEndian.PutUint64(buf[:], next)
		cfg.Put(keyNextSlot, buf[:])
	}
	return slots
}

// freeSlot records slot as reusable. Keys are big-endian so the cursor
// hands out the lowest free slot first.
func freeSlot(tx *bolt.Tx, slot uint32) error {
	var key [4]byte
	binary.BigEndian.PutUint32(key[:], slot)
	return tx.Bucket(bucketFree).Put(key[:], []byte{})
}

// migrateFreeSlots moves the legacy free list, stored as one little-endian
// blob under config/free_slots, into the free_slots bucket.
func migrateFreeSlots(tx *bolt.Tx) error {
	cfg := tx.Bucket(bucketConfig)
	raw := cfg.Get(keyFreeSlots)
	if raw == nil {
		return nil
	}
	for i := 0; i+4 <= len(raw); i += 4 {
		if err := freeSlot(tx, binary.LittleEndian.Uint32(raw[i:])); err != nil {
			return err
		}
	}
	return cfg.Delete(keyFreeSlots)
}

func putSlot(tx *bolt.Tx, id string, slot uint32) error {
	var slotBuf [4]byte
	binary.LittleEndian.PutUint32(slotBuf[:], slot)
	return tx.Bucket(bucketSlots).Put([]byte(id), slotBuf[:])
}

func (m *MetaStore) GetSlot(id string) (uint32, error) {
	var slot uint32
	err := m.db.View(func(tx *bolt.Tx) error {
		raw := tx.Bucket(bucketSlots).Get([]byte(id))
		if raw == nil {
			return ErrNotFound
		}
		slot = binary.LittleEndian.Uint32(raw)
		return nil
	})
	return slot, err
}

func encodeMeta(meta map[string]any) ([]byte, error) {
	return msgpack.Marshal(meta)
}

func (m *MetaStore) PutMeta(id string, meta map[string]any) error {
	data, err := encodeMeta(meta)
	if err != nil {
		return err
	}
	return m.db.Update(func(tx *bolt.Tx) error {
		return tx.Bucket(bucketMeta).Put([]byte(id), data)
	})
}

func (m *MetaStore) GetMeta(id string) (map[string]any, error) {
	var result map[string]any
	err := m.db.View(func(tx *bolt.Tx) error {
		raw := tx.Bucket(bucketMeta).Get([]byte(id))
		if raw == nil {
			return nil
		}
		return msgpack.Unmarshal(raw, &result)
	})
	return result, err
}

func (m *MetaStore) Delete(id string) error {
	return m.db.Update(func(tx *bolt.Tx) error {
		slotRaw := tx.Bucket(bucketSlots).Get([]byte(id))
		if slotRaw == nil {
			return ErrNotFound
		}
		slot := binary.LittleEndian.Uint32(slotRaw)

		if err := freeSlot(tx, slot); err != nil {
			return err
		}
		if err := tx.Bucket(bucketSlots).Delete([]byte(id)); err != nil {
			return err
		}
		return tx.Bucket(bucketMeta).Delete([]byte(id))
	})
}

func (m *MetaStore) AllSlots() ([]string, []uint32, error) {
	var ids []string
	var slots []uint32
	err := m.db.View(func(tx *bolt.Tx) error {
		return tx.Bucket(bucketSlots).ForEach(func(k, v []byte) error {
			ids = append(ids, string(k))
			slots = append(slots, binary.LittleEndian.Uint32(v))
			return nil
		})
	})
	return ids, slots, err
}

func (m *MetaStore) Close() error {
	return m.db.Close()
}

// SlotMeta is the result of a batched slot/metadata lookup for one id.
type SlotMeta struct {
	Slot     uint32
	Found    bool
	Metadata map[string]any
}

// GetSlotsAndMeta resolves the slot (and optionally the metadata) for every id
// inside a single read transaction. The result is aligned with ids; entries
// whose id is unknown have Found == false.
func (m *MetaStore) GetSlotsAndMeta(ids []string, withMeta bool) ([]SlotMeta, error) {
	out := make([]SlotMeta, len(ids))
	err := m.db.View(func(tx *bolt.Tx) error {
		slots := tx.Bucket(bucketSlots)
		metas := tx.Bucket(bucketMeta)
		for i, id := range ids {
			key := []byte(id)
			raw := slots.Get(key)
			if raw == nil {
				continue
			}
			out[i].Slot = binary.LittleEndian.Uint32(raw)
			out[i].Found = true
			if !withMeta {
				continue
			}
			if mraw := metas.Get(key); mraw != nil {
				if err := msgpack.Unmarshal(mraw, &out[i].Metadata); err != nil {
					return err
				}
			}
		}
		return nil
	})
	return out, err
}
