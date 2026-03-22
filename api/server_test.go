package api_test

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/fayez/goatdb/api"
	"github.com/fayez/goatdb/db"
)

func setup(t *testing.T) (http.Handler, func()) {
	t.Helper()
	database, err := db.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := api.NewServer(database, ":0")
	return srv.Handler(), func() { database.Close() }
}

func post(t *testing.T, h http.Handler, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPost, path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}

func get(t *testing.T, h http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodGet, path, nil))
	return w
}

func del(t *testing.T, h http.Handler, path string) *httptest.ResponseRecorder {
	t.Helper()
	w := httptest.NewRecorder()
	h.ServeHTTP(w, httptest.NewRequest(http.MethodDelete, path, nil))
	return w
}

func put(t *testing.T, h http.Handler, path string, body any) *httptest.ResponseRecorder {
	t.Helper()
	b, _ := json.Marshal(body)
	req := httptest.NewRequest(http.MethodPut, path, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}

// ── health ────────────────────────────────────────────────────────────────────

func TestHealth(t *testing.T) {
	h, close := setup(t)
	defer close()

	w := get(t, h, "/health")
	if w.Code != http.StatusOK {
		t.Errorf("expected 200, got %d", w.Code)
	}
}

func TestMetrics(t *testing.T) {
	h, close := setup(t)
	defer close()

	get(t, h, "/health")
	w := get(t, h, "/metrics")
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var result map[string]int64
	json.NewDecoder(w.Body).Decode(&result)
	if result["requests_total"] == 0 {
		t.Error("expected non-zero request count")
	}
}

// ── collections ───────────────────────────────────────────────────────────────

func TestCreateCollection(t *testing.T) {
	h, close := setup(t)
	defer close()

	w := post(t, h, "/collections", map[string]any{
		"name": "test", "dim": 3, "metric": "euclidean", "index_type": "flat",
	})
	if w.Code != http.StatusCreated {
		t.Errorf("expected 201, got %d: %s", w.Code, w.Body)
	}
}

func TestCreateCollectionMissingFields(t *testing.T) {
	h, close := setup(t)
	defer close()

	w := post(t, h, "/collections", map[string]any{"name": "test"})
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
	var resp map[string]string
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["code"] != "bad_request" {
		t.Errorf("expected bad_request code, got %q", resp["code"])
	}
}

func TestCreateCollectionDuplicate(t *testing.T) {
	h, close := setup(t)
	defer close()

	body := map[string]any{"name": "dup", "dim": 2, "metric": "cosine", "index_type": "flat"}
	post(t, h, "/collections", body)
	w := post(t, h, "/collections", body)
	if w.Code != http.StatusConflict {
		t.Errorf("expected 409, got %d", w.Code)
	}
}

func TestListCollections(t *testing.T) {
	h, close := setup(t)
	defer close()

	post(t, h, "/collections", map[string]any{"name": "a", "dim": 2, "metric": "euclidean"})
	post(t, h, "/collections", map[string]any{"name": "b", "dim": 2, "metric": "euclidean"})

	w := get(t, h, "/collections")
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp map[string][]string
	json.NewDecoder(w.Body).Decode(&resp)
	if len(resp["collections"]) != 2 {
		t.Errorf("expected 2 collections, got %v", resp["collections"])
	}
}

func TestGetCollection(t *testing.T) {
	h, close := setup(t)
	defer close()

	post(t, h, "/collections", map[string]any{"name": "info", "dim": 4, "metric": "cosine", "index_type": "flat"})
	w := get(t, h, "/collections/info")
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["name"] != "info" || resp["dim"] != float64(4) {
		t.Errorf("unexpected response: %v", resp)
	}
}

func TestGetCollectionNotFound(t *testing.T) {
	h, close := setup(t)
	defer close()

	w := get(t, h, "/collections/missing")
	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestDropCollection(t *testing.T) {
	h, close := setup(t)
	defer close()

	post(t, h, "/collections", map[string]any{"name": "drop-me", "dim": 2, "metric": "euclidean"})
	w := del(t, h, "/collections/drop-me")
	if w.Code != http.StatusNoContent {
		t.Errorf("expected 204, got %d", w.Code)
	}
	if get(t, h, "/collections/drop-me").Code != http.StatusNotFound {
		t.Error("expected collection to be gone")
	}
}

// ── vectors ───────────────────────────────────────────────────────────────────

func createTestCollection(t *testing.T, h http.Handler, name string) {
	t.Helper()
	post(t, h, "/collections", map[string]any{"name": name, "dim": 2, "metric": "euclidean", "index_type": "flat"})
}

func TestAddAndGetVector(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	w := post(t, h, "/collections/vecs/vectors", map[string]any{
		"id": "v1", "embeddings": []float32{1, 0}, "metadata": map[string]any{"label": "a"},
	})
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body)
	}

	w = get(t, h, "/collections/vecs/vectors/v1")
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var resp map[string]any
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["id"] != "v1" {
		t.Errorf("expected id=v1, got %v", resp["id"])
	}
}

func TestGetVectorNotFound(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	w := get(t, h, "/collections/vecs/vectors/missing")
	if w.Code != http.StatusNotFound {
		t.Errorf("expected 404, got %d", w.Code)
	}
}

func TestAddVectorDimensionMismatch(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	w := post(t, h, "/collections/vecs/vectors", map[string]any{
		"id": "v1", "embeddings": []float32{1, 0, 0},
	})
	if w.Code != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", w.Code)
	}
}

func TestUpdateVector(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	post(t, h, "/collections/vecs/vectors", map[string]any{"id": "v1", "embeddings": []float32{1, 0}})
	w := put(t, h, "/collections/vecs/vectors/v1", map[string]any{"embeddings": []float32{0, 1}})
	if w.Code != http.StatusNoContent {
		t.Errorf("expected 204, got %d", w.Code)
	}
}

func TestDeleteVector(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	post(t, h, "/collections/vecs/vectors", map[string]any{"id": "v1", "embeddings": []float32{1, 0}})
	w := del(t, h, "/collections/vecs/vectors/v1")
	if w.Code != http.StatusNoContent {
		t.Errorf("expected 204, got %d", w.Code)
	}
	if get(t, h, "/collections/vecs/vectors/v1").Code != http.StatusNotFound {
		t.Error("expected vector to be gone")
	}
}

func TestBulkInsert(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	w := post(t, h, "/collections/vecs/vectors/batch", map[string]any{
		"vectors": []map[string]any{
			{"id": "a", "embeddings": []float32{1, 0}},
			{"id": "b", "embeddings": []float32{0, 1}},
		},
	})
	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body)
	}
	var resp map[string]int
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["inserted"] != 2 {
		t.Errorf("expected inserted=2, got %v", resp)
	}
}

// ── search ────────────────────────────────────────────────────────────────────

func TestSearch(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	post(t, h, "/collections/vecs/vectors", map[string]any{"id": "a", "embeddings": []float32{1, 0}})
	post(t, h, "/collections/vecs/vectors", map[string]any{"id": "b", "embeddings": []float32{0, 1}})

	w := post(t, h, "/collections/vecs/search", map[string]any{
		"embeddings": []float32{1, 0}, "top_k": 1,
	})
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body)
	}
	var resp map[string][]map[string]any
	json.NewDecoder(w.Body).Decode(&resp)
	if len(resp["results"]) != 1 || resp["results"][0]["id"] != "a" {
		t.Errorf("expected 'a' as top result, got %v", resp["results"])
	}
}

func TestSearchSortedByDistance(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	post(t, h, "/collections/vecs/vectors", map[string]any{"id": "far", "embeddings": []float32{0, 1}})
	post(t, h, "/collections/vecs/vectors", map[string]any{"id": "near", "embeddings": []float32{1, 0}})

	w := post(t, h, "/collections/vecs/search", map[string]any{
		"embeddings": []float32{1, 0}, "top_k": 2,
	})
	var resp map[string][]map[string]any
	json.NewDecoder(w.Body).Decode(&resp)
	results := resp["results"]
	if len(results) < 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0]["id"] != "near" {
		t.Errorf("expected nearest first, got %v then %v", results[0]["id"], results[1]["id"])
	}
}
