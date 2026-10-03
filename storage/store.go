package storage

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"
)

type StoredVector struct {
	Id         string
	Embeddings []float32
	Metadata   map[string]any
}

type Store struct {
	vectors *VectorStore
	meta    *MetaStore
	wal     *WAL
	mu      sync.RWMutex
}

func Open(dir string, dim int) (*Store, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	wal, entries, err := openWAL(filepath.Join(dir, "wal.log"))
	if err != nil {
		return nil, err
	}

	vs, err := openVectorStore(filepath.Join(dir, "vectors.bin"), dim)
	if err != nil {
		return nil, err
	}

	ms, err := openMetaStore(filepath.Join(dir, "meta.db"))
	if err != nil {
		return nil, err
	}

	s := &Store{vectors: vs, meta: ms, wal: wal}

	if err := s.replay(entries); err != nil {
		return nil, err
	}

	if err := wal.Truncate(); err != nil {
		return nil, err
	}

	return s, nil
}

func (s *Store) replay(entries []walEntry) error {
	for _, e := range entries {
		switch e.op {
		case opInsert:
			if _, err := s.meta.GetSlot(e.id); err == nil {
				continue
			}
			if err := s.vectors.Write(int(e.slot), e.embeddings); err != nil {
				return err
			}
			if err := s.meta.PutSlot(e.id, e.slot); err != nil {
				return err
			}
		case opUpdate:
			if _, err := s.meta.GetSlot(e.id); err != nil {
				continue
			}
			if err := s.vectors.Write(int(e.slot), e.embeddings); err != nil {
				return err
			}
		case opDelete:
			s.vectors.Delete(int(e.slot))
			s.meta.Delete(e.id)
		}
	}
	return nil
}

func (s *Store) Add(id string, embeddings []float32, metadata map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	slot, err := s.meta.AllocSlot(id)
	if err != nil {
		return err
	}

	if err := s.wal.Append(opInsert, id, slot, embeddings); err != nil {
		return err
	}
	if err := s.wal.Sync(); err != nil {
		return err
	}

	if err := s.vectors.Write(int(slot), embeddings); err != nil {
		return err
	}

	return s.meta.PutMeta(id, metadata)
}

func (s *Store) Update(id string, embeddings []float32, metadata map[string]any) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	slot, err := s.meta.GetSlot(id)
	if err != nil {
		return fmt.Errorf("vector %q not found", id)
	}

	if err := s.wal.Append(opUpdate, id, slot, embeddings); err != nil {
		return err
	}
	if err := s.wal.Sync(); err != nil {
		return err
	}

	if err := s.vectors.Write(int(slot), embeddings); err != nil {
		return err
	}

	return s.meta.PutMeta(id, metadata)
}

func (s *Store) Get(id string) ([]float32, map[string]any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	slot, err := s.meta.GetSlot(id)
	if err != nil {
		return nil, nil, fmt.Errorf("vector %q not found", id)
	}

	emb, ok := s.vectors.Read(int(slot))
	if !ok {
		return nil, nil, fmt.Errorf("vector %q is deleted", id)
	}

	meta, err := s.meta.GetMeta(id)
	if err != nil {
		return nil, nil, err
	}

	return emb, meta, nil
}

func (s *Store) Delete(id string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	slot, err := s.meta.GetSlot(id)
	if err != nil {
		return fmt.Errorf("vector %q not found", id)
	}

	if err := s.wal.Append(opDelete, id, slot, nil); err != nil {
		return err
	}
	if err := s.wal.Sync(); err != nil {
		return err
	}

	s.vectors.Delete(int(slot))
	return s.meta.Delete(id)
}

func (s *Store) LoadAll() ([]StoredVector, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ids, slots, err := s.meta.AllSlots()
	if err != nil {
		return nil, err
	}

	result := make([]StoredVector, 0, len(ids))
	for i, id := range ids {
		emb, ok := s.vectors.Read(int(slots[i]))
		if !ok {
			continue
		}
		meta, err := s.meta.GetMeta(id)
		if err != nil {
			return nil, err
		}
		result = append(result, StoredVector{Id: id, Embeddings: emb, Metadata: meta})
	}
	return result, nil
}

func (s *Store) LoadEmbeddings() ([]StoredVector, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	ids, slots, err := s.meta.AllSlots()
	if err != nil {
		return nil, err
	}

	result := make([]StoredVector, 0, len(ids))
	for i, id := range ids {
		emb, ok := s.vectors.Read(int(slots[i]))
		if !ok {
			continue
		}
		result = append(result, StoredVector{Id: id, Embeddings: emb})
	}
	return result, nil
}

func (s *Store) Close() error {
	s.wal.Close()
	s.vectors.Close()
	return s.meta.Close()
}

// GetMany hydrates several vectors at once. Slots and metadata are resolved in
// a single metadata read transaction, and embeddings are decoded from the
// vector file only when withEmb is set. The result is aligned with ids; ids
// that are missing or deleted come back with an empty Id.
func (s *Store) GetMany(ids []string, withEmb, withMeta bool) ([]StoredVector, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	entries, err := s.meta.GetSlotsAndMeta(ids, withMeta)
	if err != nil {
		return nil, err
	}

	out := make([]StoredVector, len(ids))
	for i, e := range entries {
		if !e.Found {
			continue
		}
		if withEmb {
			emb, ok := s.vectors.Read(int(e.Slot))
			if !ok {
				continue
			}
			out[i].Embeddings = emb
		} else if !s.vectors.Live(int(e.Slot)) {
			continue
		}
		out[i].Id = ids[i]
		out[i].Metadata = e.Metadata
	}
	return out, nil
}
