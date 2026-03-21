package db

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"

	"github.com/fayez/goatdb/core"
	"github.com/fayez/goatdb/index"
)

type collectionConfig struct {
	Name      string             `json:"name"`
	Dim       int                `json:"dim"`
	Metric    core.DistanceMetric `json:"metric"`
	IndexType string             `json:"index_type"`
}

type Database struct {
	dir         string
	collections map[string]*core.Collection
	mu          sync.RWMutex
}

func Open(dir string) (*Database, error) {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return nil, err
	}

	db := &Database{
		dir:         dir,
		collections: make(map[string]*core.Collection),
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
			continue
		}
		idx, err := createIndex(cfg.IndexType, cfg.Dim, cfg.Metric)
		if err != nil {
			return nil, err
		}
		col, err := core.NewCollection(cfg.Name, cfg.Dim, dir, idx)
		if err != nil {
			return nil, err
		}
		db.collections[cfg.Name] = col
	}

	return db, nil
}

func (db *Database) CreateCollection(name string, dim int, metric core.DistanceMetric, indexType string) (*core.Collection, error) {
	db.mu.Lock()
	defer db.mu.Unlock()

	if _, exists := db.collections[name]; exists {
		return nil, fmt.Errorf("collection %q already exists", name)
	}

	colDir := filepath.Join(db.dir, name)
	if err := os.MkdirAll(colDir, 0755); err != nil {
		return nil, err
	}

	cfg := collectionConfig{Name: name, Dim: dim, Metric: metric, IndexType: indexType}
	if err := saveConfig(colDir, cfg); err != nil {
		return nil, err
	}

	idx, err := createIndex(indexType, dim, metric)
	if err != nil {
		return nil, err
	}

	col, err := core.NewCollection(name, dim, db.dir, idx)
	if err != nil {
		return nil, err
	}

	db.collections[name] = col
	return col, nil
}

func (db *Database) GetCollection(name string) (*core.Collection, error) {
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

func createIndex(indexType string, dim int, metric core.DistanceMetric) (core.Index, error) {
	switch indexType {
	case "flat":
		return index.NewFlatIndex(dim, metric), nil
	case "lsh":
		return index.NewLSHIndex(dim, 10, 8, metric), nil
	case "ivf":
		return index.NewIVFIndex(dim, 100, 10, metric), nil
	case "hnsw":
		return index.NewHNSWIndex(dim, 16, 200, 50, metric), nil
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
