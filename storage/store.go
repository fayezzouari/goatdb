package storage

import (
	"errors"
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

const walCheckpointSize = 64 << 20

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

	if err := s.checkpoint(); err != nil {
		return nil, err
	}

	return s, nil
}

// replay re-applies vector file writes from the WAL. MetaStore commits are
// atomic and durable, so it is the source of truth for which id owns which
// slot; WAL entries only re-apply embeddings and delete flags that may not
// have reached vectors.bin. Entries that never committed to MetaStore are
// skipped, which makes replay idempotent.
func (s *Store) replay(entries []walEntry) error {
	for _, e := range entries {
		slot, err := s.meta.GetSlot(e.id)
		owned := err == nil && slot == e.slot
		switch e.op {
		case opInsert, opUpdate:
			if !owned {
				continue
			}
			if err := s.vectors.Write(int(e.slot), e.embeddings); err != nil {
				return err
			}
		case opDelete:
			if owned {
				continue
			}
			s.vectors.Delete(int(e.slot))
		}
	}
	return nil
}

// checkpoint flushes the vector file and truncates the WAL.
func (s *Store) checkpoint() error {
	if err := s.vectors.Flush(); err != nil {
		return err
	}
	return s.wal.Truncate()
}

func (s *Store) maybeCheckpoint() error {
	if s.wal.Size() < walCheckpointSize {
		return nil
	}
	return s.checkpoint()
}

func (s *Store) Checkpoint() error {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.checkpoint()
}

func (s *Store) Add(id string, embeddings []float32, metadata map[string]any) error {
	return s.AddBatch([]StoredVector{{Id: id, Embeddings: embeddings, Metadata: metadata}})
}

// AddBatch inserts vectors with one WAL write, one fsync and one metadata
// transaction.
func (s *Store) AddBatch(vectors []StoredVector) error {
	if len(vectors) == 0 {
		return nil
	}
	ids := make([]string, len(vectors))
	metas := make([][]byte, len(vectors))
	for i, v := range vectors {
		if len(v.Embeddings) != s.vectors.dim {
			return fmt.Errorf("vector %q: expected %d dimensions, got %d", v.Id, s.vectors.dim, len(v.Embeddings))
		}
		ids[i] = v.Id
		m, err := encodeMeta(v.Metadata)
		if err != nil {
			return err
		}
		metas[i] = m
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.maybeCheckpoint(); err != nil {
		return err
	}

	slots, err := s.meta.Insert(ids, metas, func(slots []uint32) error {
		entries := make([]walEntry, len(vectors))
		for i, v := range vectors {
			entries[i] = walEntry{op: opInsert, id: v.Id, slot: slots[i], embeddings: v.Embeddings}
		}
		if err := s.wal.AppendBatch(entries); err != nil {
			return err
		}
		return s.wal.Sync()
	})
	if err != nil {
		return err
	}

	for i, v := range vectors {
		if err := s.vectors.Write(int(slots[i]), v.Embeddings); err != nil {
			return err
		}
	}
	return nil
}

func (s *Store) Update(id string, embeddings []float32, metadata map[string]any) error {
	if len(embeddings) != s.vectors.dim {
		return fmt.Errorf("vector %q: expected %d dimensions, got %d", id, s.vectors.dim, len(embeddings))
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if err := s.maybeCheckpoint(); err != nil {
		return err
	}

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

	if err := s.maybeCheckpoint(); err != nil {
		return err
	}

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

	if err := s.meta.Delete(id); err != nil {
		return err
	}
	s.vectors.Delete(int(slot))
	return nil
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
	s.mu.Lock()
	defer s.mu.Unlock()

	var errs []error
	if err := s.vectors.Flush(); err != nil {
		errs = append(errs, err)
	} else if err := s.wal.Truncate(); err != nil {
		errs = append(errs, err)
	}
	errs = append(errs, s.wal.Close(), s.vectors.Close(), s.meta.Close())
	return errors.Join(errs...)
}
