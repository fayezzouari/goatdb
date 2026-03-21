package db

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/fayez/goatdb/core"
	"github.com/fayez/goatdb/storage"
)

type Collection struct {
	name  string
	dim   int
	dir   string
	store *storage.Store
	index core.Index
	mu    sync.RWMutex
}

func newCollection(name string, dim int, dir string, idx core.Index) (*Collection, error) {
	store, err := storage.Open(filepath.Join(dir, name), dim)
	if err != nil {
		return nil, err
	}

	c := &Collection{
		name:  name,
		dim:   dim,
		dir:   dir,
		store: store,
		index: idx,
	}

	indexPath := filepath.Join(dir, name, "index.bin")
	if p, ok := idx.(core.Persistable); ok {
		if _, err := os.Stat(indexPath); err == nil {
			if err := p.Load(indexPath); err != nil {
				if err := c.rebuildIndex(); err != nil {
					return nil, err
				}
			}
		} else {
			if err := c.rebuildIndex(); err != nil {
				return nil, err
			}
		}
	}

	return c, nil
}

func (c *Collection) rebuildIndex() error {
	vectors, err := c.store.LoadEmbeddings()
	if err != nil {
		return err
	}
	for _, sv := range vectors {
		c.index.AddVector(sv.Id, core.Vector{Embeddings: sv.Embeddings})
	}
	return nil
}

func (c *Collection) AddVector(id string, vector core.Vector) error {
	if c.dim != len(vector.Embeddings) {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", c.dim, len(vector.Embeddings))
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.store.Add(id, vector.Embeddings, vector.Metadata); err != nil {
		return err
	}
	c.index.AddVector(id, core.Vector{Embeddings: vector.Embeddings})
	return nil
}

func (c *Collection) GetVector(id string) (core.Vector, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	emb, meta, err := c.store.Get(id)
	if err != nil {
		return core.Vector{}, false
	}
	return core.Vector{Embeddings: emb, Metadata: meta}, true
}

func (c *Collection) UpdateVector(id string, vector core.Vector) error {
	if c.dim != len(vector.Embeddings) {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", c.dim, len(vector.Embeddings))
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.store.Update(id, vector.Embeddings, vector.Metadata); err != nil {
		return err
	}
	c.index.DeleteVector(id)
	c.index.AddVector(id, core.Vector{Embeddings: vector.Embeddings})
	return nil
}

func (c *Collection) Train() error {
	t, ok := c.index.(core.Trainable)
	if !ok {
		return nil
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	svecs, err := c.store.LoadEmbeddings()
	if err != nil {
		return err
	}
	vectors := make([]core.Vector, len(svecs))
	for i, sv := range svecs {
		vectors[i] = core.Vector{Embeddings: sv.Embeddings}
	}
	t.Train(vectors)
	return nil
}

func (c *Collection) DeleteVector(id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.store.Delete(id); err != nil {
		return err
	}
	c.index.DeleteVector(id)
	return nil
}

func (c *Collection) Search(query core.Vector, topK int) ([]core.SearchResult, error) {
	if c.dim != len(query.Embeddings) {
		return nil, fmt.Errorf("dimension mismatch: expected %d, got %d", c.dim, len(query.Embeddings))
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	results := c.index.Search(query, topK)
	for i, r := range results {
		emb, meta, err := c.store.Get(r.Id)
		if err != nil {
			continue
		}
		results[i].Vector = core.Vector{Embeddings: emb, Metadata: meta}
	}
	return results, nil
}

func (c *Collection) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if p, ok := c.index.(core.Persistable); ok {
		if err := p.Save(filepath.Join(c.dir, c.name, "index.bin")); err != nil {
			return err
		}
	}
	return c.store.Close()
}
