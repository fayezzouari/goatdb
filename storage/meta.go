package storage

import (
	"encoding/binary"
	"errors"

	msgpack "github.com/shamaton/msgpack/v2"
	bolt "go.etcd.io/bbolt"
)

var (
	bucketSlots  = []byte("slots")
	bucketMeta   = []byte("meta")
	bucketConfig = []byte("config")
	keyNextSlot  = []byte("next_slot")
	keyFreeSlots = []byte("free_slots")
)

var ErrNotFound = errors.New("not found")

type MetaStore struct {
	db *bolt.DB
}

func openMetaStore(path string) (*MetaStore, error) {
	db, err := bolt.Open(path, 0644, nil)
	if err != nil {
		return nil, err
	}
	err = db.Update(func(tx *bolt.Tx) error {
		for _, b := range [][]byte{bucketSlots, bucketMeta, bucketConfig} {
			if _, err := tx.CreateBucketIfNotExists(b); err != nil {
				return err
			}
		}
		return nil
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
// transaction. beforeCommit runs inside the transaction once slots are known;
// if it fails, nothing is committed.
func (m *MetaStore) Insert(ids []string, metas [][]byte, beforeCommit func(slots []uint32) error) ([]uint32, error) {
	var slots []uint32
	err := m.db.Update(func(tx *bolt.Tx) error {
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

	if raw := cfg.Get(keyFreeSlots); len(raw) >= 4 {
		k := min(n, len(raw)/4)
		for i := 0; i < k; i++ {
			slots = append(slots, binary.LittleEndian.Uint32(raw[i*4:]))
		}
		cfg.Put(keyFreeSlots, append([]byte(nil), raw[k*4:]...))
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

		cfg := tx.Bucket(bucketConfig)
		existing := cfg.Get(keyFreeSlots)
		var freeBuf [4]byte
		binary.LittleEndian.PutUint32(freeBuf[:], slot)
		cfg.Put(keyFreeSlots, append(existing, freeBuf[:]...))

		tx.Bucket(bucketSlots).Delete([]byte(id))
		tx.Bucket(bucketMeta).Delete([]byte(id))
		return nil
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
