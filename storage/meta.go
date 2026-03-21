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
		cfg := tx.Bucket(bucketConfig)

		if raw := cfg.Get(keyFreeSlots); len(raw) >= 4 {
			slot = binary.LittleEndian.Uint32(raw[:4])
			cfg.Put(keyFreeSlots, raw[4:])
		} else {
			var next uint64
			if raw := cfg.Get(keyNextSlot); raw != nil {
				next = binary.LittleEndian.Uint64(raw)
			}
			slot = uint32(next)
			var buf [8]byte
			binary.LittleEndian.PutUint64(buf[:], next+1)
			cfg.Put(keyNextSlot, buf[:])
		}

		var slotBuf [4]byte
		binary.LittleEndian.PutUint32(slotBuf[:], slot)
		return tx.Bucket(bucketSlots).Put([]byte(id), slotBuf[:])
	})
	return slot, err
}

func (m *MetaStore) PutSlot(id string, slot uint32) error {
	return m.db.Update(func(tx *bolt.Tx) error {
		var slotBuf [4]byte
		binary.LittleEndian.PutUint32(slotBuf[:], slot)
		return tx.Bucket(bucketSlots).Put([]byte(id), slotBuf[:])
	})
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

func (m *MetaStore) PutMeta(id string, meta map[string]any) error {
	data, err := msgpack.Marshal(meta)
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
