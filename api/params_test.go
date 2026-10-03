package api_test

import (
	"encoding/json"
	"net/http"
	"testing"
)

func TestCreateCollectionWithIndexParams(t *testing.T) {
	h, close := setup(t)
	defer close()

	w := post(t, h, "/collections", map[string]any{
		"name": "hp", "dim": 2, "index_type": "hnsw",
		"hnsw": map[string]any{"m": 8, "ef_construction": 64},
	})
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body)
	}
	w = post(t, h, "/collections", map[string]any{
		"name": "ip", "dim": 2, "index_type": "ivf",
		"ivf": map[string]any{"nlist": 8, "nprobe": 2},
	})
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body)
	}

	type info struct {
		HNSW map[string]int `json:"hnsw"`
		IVF  map[string]int `json:"ivf"`
	}
	var hi info
	json.NewDecoder(get(t, h, "/collections/hp").Body).Decode(&hi)
	if hi.HNSW["m"] != 8 || hi.HNSW["ef_construction"] != 64 || hi.HNSW["ef_search"] != 128 || hi.IVF != nil {
		t.Errorf("unexpected hnsw info: %+v", hi)
	}
	var ii info
	json.NewDecoder(get(t, h, "/collections/ip").Body).Decode(&ii)
	if ii.IVF["nlist"] != 8 || ii.IVF["nprobe"] != 2 || ii.HNSW != nil {
		t.Errorf("unexpected ivf info: %+v", ii)
	}
}

func TestCreateCollectionInvalidIndexParams(t *testing.T) {
	h, close := setup(t)
	defer close()

	bodies := []map[string]any{
		{"name": "a", "dim": 2, "index_type": "hnsw", "hnsw": map[string]any{"m": -1}},
		{"name": "b", "dim": 2, "index_type": "hnsw", "hnsw": map[string]any{"ef_search": 10001}},
		{"name": "c", "dim": 2, "index_type": "hnsw", "hnsw": map[string]any{"ef_construction": -5}},
		{"name": "d", "dim": 2, "index_type": "ivf", "ivf": map[string]any{"nlist": -2}},
		{"name": "e", "dim": 2, "index_type": "ivf", "ivf": map[string]any{"nlist": 4, "nprobe": 9}},
	}
	for _, b := range bodies {
		if w := post(t, h, "/collections", b); w.Code != http.StatusBadRequest {
			t.Errorf("%v: expected 400, got %d: %s", b, w.Code, w.Body)
		}
	}
}

func TestSearchWithEf(t *testing.T) {
	h, close := setup(t)
	defer close()

	for _, typ := range []string{"hnsw", "ivf", "flat"} {
		post(t, h, "/collections", map[string]any{"name": typ, "dim": 2, "index_type": typ})
		post(t, h, "/collections/"+typ+"/vectors", map[string]any{"id": "a", "embeddings": []float32{1, 0}})
		post(t, h, "/collections/"+typ+"/vectors", map[string]any{"id": "b", "embeddings": []float32{0, 1}})

		w := post(t, h, "/collections/"+typ+"/search", map[string]any{
			"embeddings": []float32{1, 0}, "top_k": 1, "ef": 200,
		})
		if w.Code != http.StatusOK {
			t.Fatalf("%s: expected 200, got %d: %s", typ, w.Code, w.Body)
		}
		var resp map[string][]map[string]any
		json.NewDecoder(w.Body).Decode(&resp)
		if len(resp["results"]) != 1 || resp["results"][0]["id"] != "a" {
			t.Errorf("%s: expected 'a', got %v", typ, resp["results"])
		}
	}
}

func TestSearchInvalidEf(t *testing.T) {
	h, close := setup(t)
	defer close()
	post(t, h, "/collections", map[string]any{"name": "v", "dim": 2, "index_type": "hnsw"})
	post(t, h, "/collections/v/vectors", map[string]any{"id": "a", "embeddings": []float32{1, 0}})

	for _, ef := range []int{0, -1, 10001} {
		w := post(t, h, "/collections/v/search", map[string]any{"embeddings": []float32{1, 0}, "ef": ef})
		if w.Code != http.StatusBadRequest {
			t.Errorf("ef=%d: expected 400, got %d: %s", ef, w.Code, w.Body)
		}
	}
}
