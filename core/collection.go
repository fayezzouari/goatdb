package core

import (
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/fayez/goatdb/storage"
)

type DistanceMetric string

const (
	Cosine     DistanceMetric = "cosine"
	Euclidean  DistanceMetric = "euclidean"
	DotProduct DistanceMetric = "dot_product"
	Manhattan  DistanceMetric = "manhattan"
)

type SearchResult struct {
	Id       string
	Distance float32
	Vector   Vector
}

type Collection struct {
	name  string
	dim   int
	dir   string
	store *storage.Store
	index Index
	mu    sync.RWMutex
}

func NewCollection(name string, dim int, dir string, idx Index) (*Collection, error) {
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
	if _, err := os.Stat(indexPath); err == nil {
		if err := idx.Load(indexPath); err != nil {
			if err := c.rebuildIndex(); err != nil {
				return nil, err
			}
		}
	} else {
		if err := c.rebuildIndex(); err != nil {
			return nil, err
		}
	}

	return c, nil
}

func (c *Collection) rebuildIndex() error {
	vectors, err := c.store.LoadAll()
	if err != nil {
		return err
	}
	for _, sv := range vectors {
		c.index.AddVector(c.name, sv.Id, Vector{Embeddings: sv.Embeddings, Metadata: sv.Metadata})
	}
	return nil
}

func (c *Collection) AddVector(id string, vector Vector) error {
	if c.dim != len(vector.Embeddings) {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", c.dim, len(vector.Embeddings))
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.store.Add(id, vector.Embeddings, vector.Metadata); err != nil {
		return err
	}
	c.index.AddVector(c.name, id, vector)
	return nil
}

func (c *Collection) GetVector(id string) (Vector, bool) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	emb, meta, err := c.store.Get(id)
	if err != nil {
		return Vector{}, false
	}
	return Vector{Embeddings: emb, Metadata: meta}, true
}

func (c *Collection) DeleteVector(id string) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.store.Delete(id); err != nil {
		return err
	}
	c.index.DeleteVector(c.name, id)
	return nil
}

func (c *Collection) Search(query Vector, topK int) ([]SearchResult, error) {
	if c.dim != len(query.Embeddings) {
		return nil, fmt.Errorf("dimension mismatch: expected %d, got %d", c.dim, len(query.Embeddings))
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	return c.index.Search(c.name, query, topK), nil
}

func (c *Collection) Close() error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.index.Save(filepath.Join(c.dir, c.name, "index.bin")); err != nil {
		return err
	}
	return c.store.Close()
}
