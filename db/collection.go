package db

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"

	"github.com/fayez/goatdb/core"
	"github.com/fayez/goatdb/storage"
)

type Collection struct {
	name      string
	dim       int
	metric    core.DistanceMetric
	indexType string
	dir       string
	store     *storage.Store
	index     core.Index
	mu        sync.RWMutex
}

type CollectionInfo struct {
	Name      string              `json:"name"`
	Dim       int                 `json:"dim"`
	Metric    core.DistanceMetric `json:"metric"`
	IndexType string              `json:"index_type"`
}

func newCollection(name string, dim int, metric core.DistanceMetric, indexType string, dir string, idx core.Index) (*Collection, error) {
	store, err := storage.Open(filepath.Join(dir, name), dim)
	if err != nil {
		return nil, err
	}

	c := &Collection{
		name:      name,
		dim:       dim,
		metric:    metric,
		indexType: indexType,
		dir:       dir,
		store:     store,
		index:     idx,
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

func (c *Collection) Info() CollectionInfo {
	return CollectionInfo{Name: c.name, Dim: c.dim, Metric: c.metric, IndexType: c.indexType}
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

func (c *Collection) AddVector(ctx context.Context, id string, vector core.Vector) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(vector.Embeddings) != c.dim {
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

func (c *Collection) AddVectors(ctx context.Context, vectors map[string]core.Vector) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	batch := make([]storage.StoredVector, 0, len(vectors))
	for id, v := range vectors {
		if len(v.Embeddings) != c.dim {
			return fmt.Errorf("vector %q: dimension mismatch: expected %d, got %d", id, c.dim, len(v.Embeddings))
		}
		batch = append(batch, storage.StoredVector{Id: id, Embeddings: v.Embeddings, Metadata: v.Metadata})
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.store.AddBatch(batch); err != nil {
		return err
	}
	for _, v := range batch {
		c.index.AddVector(v.Id, core.Vector{Embeddings: v.Embeddings})
	}
	return nil
}

func (c *Collection) GetVector(ctx context.Context, id string) (core.Vector, bool) {
	if err := ctx.Err(); err != nil {
		return core.Vector{}, false
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	emb, meta, err := c.store.Get(id)
	if err != nil {
		return core.Vector{}, false
	}
	return core.Vector{Embeddings: emb, Metadata: meta}, true
}

func (c *Collection) UpdateVector(ctx context.Context, id string, vector core.Vector) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(vector.Embeddings) != c.dim {
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

func (c *Collection) DeleteVector(ctx context.Context, id string) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.store.Delete(id); err != nil {
		return err
	}
	c.index.DeleteVector(id)
	return nil
}

func (c *Collection) Search(ctx context.Context, query core.Vector, topK int) ([]core.SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(query.Embeddings) != c.dim {
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
	sort.Slice(results, func(i, j int) bool {
		return results[i].Distance < results[j].Distance
	})
	return results, nil
}

func (c *Collection) Train(ctx context.Context) error {
	if err := ctx.Err(); err != nil {
		return err
	}
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
