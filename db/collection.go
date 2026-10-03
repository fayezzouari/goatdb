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

var (
	ErrExists      = storage.ErrExists
	ErrDuplicateID = storage.ErrDuplicateID
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

// VectorEntry is a vector with its id, as passed to AddVectors.
type VectorEntry struct {
	Id string
	core.Vector
}

// AddVectors inserts all vectors or none. It fails with ErrDuplicateID if an id
// repeats within the batch and with ErrExists if an id is already stored.
func (c *Collection) AddVectors(ctx context.Context, vectors []VectorEntry) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	batch := make([]storage.StoredVector, 0, len(vectors))
	seen := make(map[string]struct{}, len(vectors))
	for _, v := range vectors {
		if len(v.Embeddings) != c.dim {
			return fmt.Errorf("vector %q: dimension mismatch: expected %d, got %d", v.Id, c.dim, len(v.Embeddings))
		}
		if _, dup := seen[v.Id]; dup {
			return fmt.Errorf("vector %q: %w", v.Id, ErrDuplicateID)
		}
		seen[v.Id] = struct{}{}
		batch = append(batch, storage.StoredVector{Id: v.Id, Embeddings: v.Embeddings, Metadata: v.Metadata})
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

// SearchOptions controls how search results are hydrated from storage.
type SearchOptions struct {
	IncludeVectors  bool
	IncludeMetadata bool
}

// Search returns the topK nearest vectors with embeddings and metadata attached.
func (c *Collection) Search(ctx context.Context, query core.Vector, topK int) ([]core.SearchResult, error) {
	return c.SearchWithOptions(ctx, query, topK, SearchOptions{IncludeVectors: true, IncludeMetadata: true})
}

// SearchWithOptions returns the topK nearest vectors, hydrating only the parts
// requested in opts.
func (c *Collection) SearchWithOptions(ctx context.Context, query core.Vector, topK int, opts SearchOptions) ([]core.SearchResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if len(query.Embeddings) != c.dim {
		return nil, fmt.Errorf("dimension mismatch: expected %d, got %d", c.dim, len(query.Embeddings))
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	results := c.index.Search(query, topK)
	if opts.IncludeVectors || opts.IncludeMetadata {
		ids := make([]string, len(results))
		for i, r := range results {
			ids[i] = r.Id
		}
		stored, err := c.store.GetMany(ids, opts.IncludeVectors, opts.IncludeMetadata)
		if err != nil {
			return nil, err
		}
		for i, sv := range stored {
			if sv.Id == "" {
				continue
			}
			results[i].Vector = core.Vector{Embeddings: sv.Embeddings, Metadata: sv.Metadata}
		}
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
	// Only reading the store needs the collection lock. The index does its
	// own locking, so training must not block searches and writes.
	c.mu.RLock()
	svecs, err := c.store.LoadEmbeddings()
	c.mu.RUnlock()
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
