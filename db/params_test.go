package db

import (
	"errors"
	"os"
	"path/filepath"
	"testing"

	"github.com/fayezzouari/goatdb/core"
)

func TestCreateCollectionWithOptionsPersists(t *testing.T) {
	dir := t.TempDir()
	d, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	_, err = d.CreateCollectionWithOptions("h", 4, core.Euclidean, "hnsw", CollectionOptions{
		HNSW: &HNSWParams{M: 8, EfSearch: 40},
	})
	if err != nil {
		t.Fatal(err)
	}
	_, err = d.CreateCollectionWithOptions("i", 4, core.Euclidean, "ivf", CollectionOptions{
		IVF: &IVFParams{NList: 16, NProbe: 4},
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := d.Close(); err != nil {
		t.Fatal(err)
	}

	d, err = Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	h, err := d.GetCollection("h")
	if err != nil {
		t.Fatal(err)
	}
	want := HNSWParams{M: 8, EfConstruction: DefaultHNSWEfConstruction, EfSearch: 40}
	if info := h.Info(); info.HNSW == nil || *info.HNSW != want || info.IVF != nil {
		t.Errorf("hnsw info after reopen = %+v, want hnsw %+v", info, want)
	}
	i, err := d.GetCollection("i")
	if err != nil {
		t.Fatal(err)
	}
	if info := i.Info(); info.IVF == nil || *info.IVF != (IVFParams{NList: 16, NProbe: 4}) || info.HNSW != nil {
		t.Errorf("ivf info after reopen = %+v", info)
	}
}

func TestLegacyConfigLoadsDefaults(t *testing.T) {
	dir := t.TempDir()
	for name, typ := range map[string]string{"h": "hnsw", "i": "ivf"} {
		colDir := filepath.Join(dir, name)
		if err := os.MkdirAll(colDir, 0755); err != nil {
			t.Fatal(err)
		}
		cfg := `{"name":"` + name + `","dim":3,"metric":"euclidean","index_type":"` + typ + `"}`
		if err := os.WriteFile(filepath.Join(colDir, "config.json"), []byte(cfg), 0644); err != nil {
			t.Fatal(err)
		}
	}
	d, err := Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()

	h, err := d.GetCollection("h")
	if err != nil {
		t.Fatal(err)
	}
	want := HNSWParams{M: DefaultHNSWM, EfConstruction: DefaultHNSWEfConstruction, EfSearch: DefaultHNSWEfSearch}
	if info := h.Info(); info.HNSW == nil || *info.HNSW != want {
		t.Errorf("legacy hnsw info = %+v, want %+v", info.HNSW, want)
	}
	i, err := d.GetCollection("i")
	if err != nil {
		t.Fatal(err)
	}
	if info := i.Info(); info.IVF == nil || *info.IVF != (IVFParams{NList: DefaultIVFNList, NProbe: DefaultIVFNProbe}) {
		t.Errorf("legacy ivf info = %+v", info.IVF)
	}
}

func TestCreateCollectionInvalidOptions(t *testing.T) {
	d, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	cases := map[string]struct {
		typ  string
		opts CollectionOptions
	}{
		"m=1":            {"hnsw", CollectionOptions{HNSW: &HNSWParams{M: 1}}},
		"negative m":     {"hnsw", CollectionOptions{HNSW: &HNSWParams{M: -4}}},
		"huge ef":        {"hnsw", CollectionOptions{HNSW: &HNSWParams{EfSearch: MaxEf + 1}}},
		"negative nlist": {"ivf", CollectionOptions{IVF: &IVFParams{NList: -1}}},
		"nprobe>nlist":   {"ivf", CollectionOptions{IVF: &IVFParams{NList: 4, NProbe: 5}}},
	}
	for name, c := range cases {
		if _, err := d.CreateCollectionWithOptions(name, 2, core.Euclidean, c.typ, c.opts); !errors.Is(err, ErrInvalidParams) {
			t.Errorf("%s: err = %v, want ErrInvalidParams", name, err)
		}
	}
	if len(d.ListCollections()) != 0 {
		t.Errorf("invalid creates left collections: %v", d.ListCollections())
	}
}

func TestSearchWithEf(t *testing.T) {
	d, err := Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	defer d.Close()
	for _, typ := range []string{"flat", "lsh", "hnsw", "ivf"} {
		col, err := d.CreateCollection(typ, 2, core.Euclidean, typ)
		if err != nil {
			t.Fatal(err)
		}
		for i, e := range [][]float32{{0, 0}, {1, 0}, {0, 1}} {
			if err := col.AddVector(ctx, string(rune('a'+i)), core.Vector{Embeddings: e}); err != nil {
				t.Fatal(err)
			}
		}
		q := core.Vector{Embeddings: []float32{0, 0}}
		res, err := col.SearchWithOptions(ctx, q, 1, SearchOptions{Ef: 50})
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		if len(res) != 1 || res[0].Id != "a" {
			t.Errorf("%s: got %v, want a", typ, res)
		}
		if _, err := col.SearchWithOptions(ctx, q, 1, SearchOptions{Ef: MaxEf + 1}); !errors.Is(err, ErrInvalidParams) {
			t.Errorf("%s: ef above max: err = %v", typ, err)
		}
	}
}
