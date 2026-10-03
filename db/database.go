package db

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"

	"github.com/fayezzouari/goatdb/core"
	"github.com/fayezzouari/goatdb/index"
)

type collectionConfig struct {
	Name      string              `json:"name"`
	Dim       int                 `json:"dim"`
	Metric    core.DistanceMetric `json:"metric"`
	IndexType string              `json:"index_type"`
	HNSW      *HNSWParams         `json:"hnsw,omitempty"`
	IVF       *IVFParams          `json:"ivf,omitempty"`
}

func (cfg collectionConfig) options() CollectionOptions {
	return CollectionOptions{HNSW: cfg.HNSW, IVF: cfg.IVF}
}

type Database struct {
	dir         string
	collections map[string]*Collection
	mu          sync.RWMutex
}

func Open(dir string) (*Database, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	db := &Database{
		dir:         dir,
		collections: make(map[string]*Collection),
	}

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}
		cfg, err := loadConfig(filepath.Join(dir, entry.Name()))
		if err != nil {
			log.Printf("warn: skipping collection dir %q: %v", entry.Name(), err)
			continue
		}
		// Configs written before index parameters were stored have none, so
		// they resolve to the defaults those collections were built with.
		opts, err := resolveOptions(cfg.IndexType, cfg.options())
		if err != nil {
			log.Printf("warn: skipping collection dir %q: %v", entry.Name(), err)
			continue
		}
		idx, err := createIndex(cfg.IndexType, cfg.Dim, cfg.Metric, opts)
		if err != nil {
			return nil, err
		}
		col, err := newCollection(cfg.Name, cfg.Dim, cfg.Metric, cfg.IndexType, dir, idx)
		if err != nil {
			return nil, err
		}
		col.opts = opts
		db.collections[cfg.Name] = col
	}

	return db, nil
}

// CreateCollection creates a collection with the default index parameters.
func (db *Database) CreateCollection(name string, dim int, metric core.DistanceMetric, indexType string) (*Collection, error) {
	return db.CreateCollectionWithOptions(name, dim, metric, indexType, CollectionOptions{})
}

// CreateCollectionWithOptions creates a collection with the given index
// parameters. Unset fields take their defaults and parameters for other index
// types are ignored. Out-of-range values fail with ErrInvalidParams.
func (db *Database) CreateCollectionWithOptions(name string, dim int, metric core.DistanceMetric, indexType string, opts CollectionOptions) (*Collection, error) {
	opts, err := resolveOptions(indexType, opts)
	if err != nil {
		return nil, err
	}
	idx, err := createIndex(indexType, dim, metric, opts)
	if err != nil {
		return nil, err
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	if _, exists := db.collections[name]; exists {
		return nil, fmt.Errorf("collection %q already exists", name)
	}

	colDir := filepath.Join(db.dir, name)
	if err := os.MkdirAll(colDir, 0755); err != nil {
		return nil, err
	}

	cfg := collectionConfig{Name: name, Dim: dim, Metric: metric, IndexType: indexType, HNSW: opts.HNSW, IVF: opts.IVF}
	if err := saveConfig(colDir, cfg); err != nil {
		return nil, err
	}

	col, err := newCollection(name, dim, metric, indexType, db.dir, idx)
	if err != nil {
		return nil, err
	}
	col.opts = opts

	db.collections[name] = col
	return col, nil
}

func (db *Database) ListCollections() []string {
	db.mu.RLock()
	defer db.mu.RUnlock()

	names := make([]string, 0, len(db.collections))
	for name := range db.collections {
		names = append(names, name)
	}
	return names
}

func (db *Database) GetCollection(name string) (*Collection, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	col, ok := db.collections[name]
	if !ok {
		return nil, fmt.Errorf("collection %q not found", name)
	}
	return col, nil
}

func (db *Database) DropCollection(name string) error {
	db.mu.Lock()
	defer db.mu.Unlock()

	col, ok := db.collections[name]
	if !ok {
		return fmt.Errorf("collection %q not found", name)
	}
	col.Close()
	delete(db.collections, name)
	return os.RemoveAll(filepath.Join(db.dir, name))
}

func (db *Database) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	for _, col := range db.collections {
		if err := col.Close(); err != nil {
			return err
		}
	}
	return nil
}

// createIndex builds an index; opts must come from resolveOptions.
func createIndex(indexType string, dim int, metric core.DistanceMetric, opts CollectionOptions) (core.Index, error) {
	switch indexType {
	case "flat":
		return index.NewFlatIndex(dim, metric), nil
	case "lsh":
		return index.NewLSHIndex(dim, defaultLSHTables, defaultLSHBits, metric), nil
	case "ivf":
		return index.NewIVFIndex(dim, opts.IVF.NList, opts.IVF.NProbe, metric), nil
	case "hnsw":
		p := opts.HNSW
		return index.NewHNSWIndex(dim, p.M, p.EfConstruction, p.EfSearch, metric), nil
	default:
		return nil, fmt.Errorf("unknown index type: %q", indexType)
	}
}

func saveConfig(dir string, cfg collectionConfig) error {
	data, err := json.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(filepath.Join(dir, "config.json"), data, 0644)
}

func loadConfig(dir string) (collectionConfig, error) {
	data, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return collectionConfig{}, err
	}
	var cfg collectionConfig
	return cfg, json.Unmarshal(data, &cfg)
}
