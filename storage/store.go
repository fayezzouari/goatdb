package storage

import (
	"fmt"
	"os"
	"path/filepath"
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
		case opDelete:
			s.vectors.Delete(int(e.slot))
			s.meta.Delete(e.id)
		}
	}
	return nil
}

func (s *Store) Add(id string, embeddings []float32, metadata map[string]any) error {
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

func (s *Store) Get(id string) ([]float32, map[string]any, error) {
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
		result = append(result, StoredVector{
			Id:         id,
			Embeddings: emb,
			Metadata:   meta,
		})
	}
	return result, nil
}

func (s *Store) Close() error {
	s.wal.Close()
	s.vectors.Close()
	return s.meta.Close()
}
