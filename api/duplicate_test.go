package api_test

import (
	"encoding/json"
	"net/http"
	"testing"
)

func TestAddVectorExistingID(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	post(t, h, "/collections/vecs/vectors", map[string]any{"id": "v1", "embeddings": []float32{1, 0}})
	w := post(t, h, "/collections/vecs/vectors", map[string]any{"id": "v1", "embeddings": []float32{0, 1}})
	if w.Code != http.StatusConflict {
		t.Fatalf("expected 409, got %d: %s", w.Code, w.Body)
	}
	var resp map[string]string
	json.NewDecoder(w.Body).Decode(&resp)
	if resp["code"] != "conflict" {
		t.Errorf("expected code=conflict, got %v", resp)
	}
}

func TestBulkInsertExistingID(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	post(t, h, "/collections/vecs/vectors", map[string]any{"id": "a", "embeddings": []float32{1, 0}})
	w := post(t, h, "/collections/vecs/vectors/batch", map[string]any{
		"vectors": []map[string]any{
			{"id": "b", "embeddings": []float32{0, 1}},
			{"id": "a", "embeddings": []float32{1, 1}},
		},
	})
	if w.Code != http.StatusConflict {
		t.Fatalf("expected 409, got %d: %s", w.Code, w.Body)
	}
	if get(t, h, "/collections/vecs/vectors/b").Code != http.StatusNotFound {
		t.Error("expected no vector from a rejected batch to be written")
	}
}

func TestBulkInsertDuplicateIDInRequest(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	w := post(t, h, "/collections/vecs/vectors/batch", map[string]any{
		"vectors": []map[string]any{
			{"id": "a", "embeddings": []float32{1, 0}},
			{"id": "a", "embeddings": []float32{0, 1}},
		},
	})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body)
	}
	if get(t, h, "/collections/vecs/vectors/a").Code != http.StatusNotFound {
		t.Error("expected nothing to be written")
	}
}

func TestBulkInsertDimensionMismatchWritesNothing(t *testing.T) {
	h, close := setup(t)
	defer close()
	createTestCollection(t, h, "vecs")

	w := post(t, h, "/collections/vecs/vectors/batch", map[string]any{
		"vectors": []map[string]any{
			{"id": "a", "embeddings": []float32{1, 0}},
			{"id": "b", "embeddings": []float32{1, 0, 0}},
		},
	})
	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d: %s", w.Code, w.Body)
	}
	if get(t, h, "/collections/vecs/vectors/a").Code != http.StatusNotFound {
		t.Error("expected nothing to be written")
	}
}
