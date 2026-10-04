package db

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"

	"github.com/fayezzouari/goatdb/core"
	"github.com/fayezzouari/goatdb/storage"
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
	// opts holds the resolved index parameters. It is set once at creation.
	opts CollectionOptions
	// writeMu serializes writes. Inserts hold it with mu read-locked, so
	// searches keep running while a batch is indexed; updates and deletes
	// hold it with mu write-locked. Close takes mu for writing.
	writeMu sync.Mutex
	mu      sync.RWMutex
}

type CollectionInfo struct {
	Name      string              `json:"name"`
	Dim       int                 `json:"dim"`
	Metric    core.DistanceMetric `json:"metric"`
	IndexType string              `json:"index_type"`
	HNSW      *HNSWParams         `json:"hnsw,omitempty"`
	IVF       *IVFParams          `json:"ivf,omitempty"`
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
	info := CollectionInfo{Name: c.name, Dim: c.dim, Metric: c.metric, IndexType: c.indexType}
	if c.opts.HNSW != nil {
		p := *c.opts.HNSW
		info.HNSW = &p
	}
	if c.opts.IVF != nil {
		p := *c.opts.IVF
		info.IVF = &p
	}
	return info
}

func (c *Collection) rebuildIndex() error {
	vectors, err := c.store.LoadEmbeddings()
	if err != nil {
		return err
	}
	addToIndex(c.index, vectors)
	return nil
}

// minVectorsPerWorker keeps small batches on one goroutine, where starting
// workers would cost more than it saves.
const minVectorsPerWorker = 64

// addToIndex inserts vectors into idx, with up to GOMAXPROCS workers when
// the index supports concurrent inserts.
func addToIndex(idx core.Index, vectors []storage.StoredVector) {
	workers := 1
	if ca, ok := idx.(core.ConcurrentAdder); ok && ca.ConcurrentAdd() {
		workers = min(runtime.GOMAXPROCS(0), len(vectors)/minVectorsPerWorker)
	}
	if workers <= 1 {
		for _, v := range vectors {
			idx.AddVector(v.Id, core.Vector{Embeddings: v.Embeddings})
		}
		return
	}
	var next atomic.Int64
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1) - 1)
				if i >= len(vectors) {
					return
				}
				v := vectors[i]
				idx.AddVector(v.Id, core.Vector{Embeddings: v.Embeddings})
			}
		}()
	}
	wg.Wait()
}

func (c *Collection) AddVector(ctx context.Context, id string, vector core.Vector) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if len(vector.Embeddings) != c.dim {
		return fmt.Errorf("dimension mismatch: expected %d, got %d", c.dim, len(vector.Embeddings))
	}
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	c.mu.RLock()
	defer c.mu.RUnlock()

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
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	c.mu.RLock()
	defer c.mu.RUnlock()

	if err := c.store.AddBatch(batch); err != nil {
		return err
	}
	// The batch is durable; index it in parallel. Searches may run
	// meanwhile and see part of the batch.
	addToIndex(c.index, batch)
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
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
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
	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	c.mu.Lock()
	defer c.mu.Unlock()

	if err := c.store.Delete(id); err != nil {
		return err
	}
	c.index.DeleteVector(id)
	return nil
}

// SearchOptions controls search depth and how results are hydrated from storage.
type SearchOptions struct {
	IncludeVectors  bool
	IncludeMetadata bool
	// Ef overrides the search depth for this query when > 0. HNSW searches
	// with beam width max(Ef, topK); IVF probes Ef lists. Flat and LSH
	// indexes ignore it. Values above MaxEf fail with ErrInvalidParams.
	Ef int
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
	if opts.Ef != 0 {
		if err := ValidateEf(opts.Ef); err != nil {
			return nil, err
		}
	}
	c.mu.RLock()
	defer c.mu.RUnlock()

	var results []core.SearchResult
	if es, ok := c.index.(core.EfSearcher); ok && opts.Ef > 0 {
		results = es.SearchEf(query, topK, opts.Ef)
	} else {
		results = c.index.Search(query, topK)
	}
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
