package tests_test

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/fayezzouari/goatdb/api"
	"github.com/fayezzouari/goatdb/db"
)

func startServer(t *testing.T) *httptest.Server {
	t.Helper()
	database, err := db.Open(t.TempDir())
	if err != nil {
		t.Fatal(err)
	}
	srv := api.NewServer(database, ":0")
	ts := httptest.NewServer(srv.Handler())
	t.Cleanup(func() {
		ts.Close()
		database.Close()
	})
	return ts
}

func do(t *testing.T, method, url string, body any) *http.Response {
	t.Helper()
	var b []byte
	if body != nil {
		b, _ = json.Marshal(body)
	}
	req, _ := http.NewRequest(method, url, bytes.NewReader(b))
	req.Header.Set("Content-Type", "application/json")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatalf("%s %s: %v", method, url, err)
	}
	return resp
}

func decode(t *testing.T, resp *http.Response, dst any) {
	t.Helper()
	defer resp.Body.Close()
	json.NewDecoder(resp.Body).Decode(dst)
}

// TestFullFlow exercises the complete lifecycle over HTTP:
// create collection → add vectors → search → update → delete → drop
func TestFullFlow(t *testing.T) {
	ts := startServer(t)
	base := ts.URL

	// Create collection
	resp := do(t, http.MethodPost, base+"/collections", map[string]any{
		"name": "items", "dim": 3, "metric": "euclidean", "index_type": "flat",
	})
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("create collection: expected 201, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Verify it appears in list
	resp = do(t, http.MethodGet, base+"/collections", nil)
	var listResp map[string][]string
	decode(t, resp, &listResp)
	if len(listResp["collections"]) != 1 || listResp["collections"][0] != "items" {
		t.Fatalf("list collections: expected [items], got %v", listResp["collections"])
	}

	// Get collection info
	resp = do(t, http.MethodGet, base+"/collections/items", nil)
	var info map[string]any
	decode(t, resp, &info)
	if info["dim"] != float64(3) {
		t.Errorf("collection dim: expected 3, got %v", info["dim"])
	}

	// Bulk insert vectors
	resp = do(t, http.MethodPost, base+"/collections/items/vectors/batch", map[string]any{
		"vectors": []map[string]any{
			{"id": "a", "embeddings": []float32{1, 0, 0}, "metadata": map[string]any{"label": "A"}},
			{"id": "b", "embeddings": []float32{0, 1, 0}, "metadata": map[string]any{"label": "B"}},
			{"id": "c", "embeddings": []float32{0, 0, 1}, "metadata": map[string]any{"label": "C"}},
		},
	})
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("bulk insert: expected 201, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Add a single vector
	resp = do(t, http.MethodPost, base+"/collections/items/vectors", map[string]any{
		"id": "d", "embeddings": []float32{1, 1, 0},
	})
	if resp.StatusCode != http.StatusCreated {
		t.Fatalf("add vector: expected 201, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Get a specific vector
	resp = do(t, http.MethodGet, base+"/collections/items/vectors/a", nil)
	var vec map[string]any
	decode(t, resp, &vec)
	if vec["id"] != "a" {
		t.Errorf("get vector: expected id=a, got %v", vec["id"])
	}

	// Search — nearest to {1,0,0} should be "a"
	resp = do(t, http.MethodPost, base+"/collections/items/search", map[string]any{
		"embeddings": []float32{1, 0, 0}, "top_k": 2,
	})
	var searchResp map[string][]map[string]any
	decode(t, resp, &searchResp)
	results := searchResp["results"]
	if len(results) == 0 || results[0]["id"] != "a" {
		t.Errorf("search: expected 'a' first, got %v", results)
	}
	// Results must be sorted by distance
	if len(results) > 1 {
		d0 := results[0]["distance"].(float64)
		d1 := results[1]["distance"].(float64)
		if d0 > d1 {
			t.Errorf("results not sorted: %v > %v", d0, d1)
		}
	}

	// Update vector "d"
	resp = do(t, http.MethodPut, base+"/collections/items/vectors/d", map[string]any{
		"embeddings": []float32{0, 0, 0},
	})
	if resp.StatusCode != http.StatusNoContent {
		t.Errorf("update vector: expected 204, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Delete vector "c"
	resp = do(t, http.MethodDelete, base+"/collections/items/vectors/c", nil)
	if resp.StatusCode != http.StatusNoContent {
		t.Errorf("delete vector: expected 204, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Confirm "c" is gone
	resp = do(t, http.MethodGet, base+"/collections/items/vectors/c", nil)
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected 404 for deleted vector, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Drop collection
	resp = do(t, http.MethodDelete, base+"/collections/items", nil)
	if resp.StatusCode != http.StatusNoContent {
		t.Errorf("drop collection: expected 204, got %d", resp.StatusCode)
	}
	resp.Body.Close()

	// Confirm gone
	resp = do(t, http.MethodGet, base+"/collections/items", nil)
	if resp.StatusCode != http.StatusNotFound {
		t.Errorf("expected 404 after drop, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestErrorResponses(t *testing.T) {
	ts := startServer(t)
	base := ts.URL

	cases := []struct {
		name   string
		method string
		path   string
		body   any
		expect int
		code   string
	}{
		{"missing collection", http.MethodGet, "/collections/nope", nil, http.StatusNotFound, "not_found"},
		{"create no dim", http.MethodPost, "/collections", map[string]any{"name": "x"}, http.StatusBadRequest, "bad_request"},
		{"vector in missing collection", http.MethodGet, "/collections/nope/vectors/v1", nil, http.StatusNotFound, "not_found"},
		{"search missing collection", http.MethodPost, "/collections/nope/search",
			map[string]any{"embeddings": []float32{1, 0}}, http.StatusNotFound, "not_found"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			resp := do(t, tc.method, base+tc.path, tc.body)
			if resp.StatusCode != tc.expect {
				t.Errorf("expected %d, got %d", tc.expect, resp.StatusCode)
			}
			var body map[string]string
			decode(t, resp, &body)
			if body["code"] != tc.code {
				t.Errorf("expected code=%q, got %q", tc.code, body["code"])
			}
		})
	}
}

func TestMetricsAccumulate(t *testing.T) {
	ts := startServer(t)
	base := ts.URL

	// Make a few requests
	do(t, http.MethodGet, base+"/health", nil).Body.Close()
	do(t, http.MethodGet, base+"/collections/missing", nil).Body.Close()

	resp := do(t, http.MethodGet, base+"/metrics", nil)
	var m map[string]int64
	decode(t, resp, &m)

	// /metrics reads counts before its own request is recorded, so we see 2
	if m["requests_total"] < 2 {
		t.Errorf("expected at least 2 completed requests, got %d", m["requests_total"])
	}
	if m["status_4xx"] < 1 {
		t.Errorf("expected at least 1 4xx, got %d", m["status_4xx"])
	}
}

func TestDuplicateCollection(t *testing.T) {
	ts := startServer(t)
	base := ts.URL

	body := map[string]any{"name": "dup", "dim": 2, "metric": "cosine"}
	do(t, http.MethodPost, base+"/collections", body).Body.Close()
	resp := do(t, http.MethodPost, base+"/collections", body)
	if resp.StatusCode != http.StatusConflict {
		t.Errorf("expected 409, got %d", resp.StatusCode)
	}
	var errResp map[string]string
	decode(t, resp, &errResp)
	if errResp["code"] != "conflict" {
		t.Errorf("expected conflict code, got %q", errResp["code"])
	}
}

func TestBulkInsertDimensionMismatch(t *testing.T) {
	ts := startServer(t)
	base := ts.URL

	do(t, http.MethodPost, base+"/collections", map[string]any{
		"name": "vecs", "dim": 2, "metric": "euclidean",
	}).Body.Close()

	resp := do(t, http.MethodPost, base+"/collections/vecs/vectors/batch", map[string]any{
		"vectors": []map[string]any{
			{"id": "ok", "embeddings": []float32{1, 0}},
			{"id": "bad", "embeddings": []float32{1, 0, 0}}, // wrong dim
		},
	})
	if resp.StatusCode != http.StatusBadRequest {
		t.Errorf("expected 400, got %d", resp.StatusCode)
	}
	resp.Body.Close()
}

func TestDefaultTopK(t *testing.T) {
	ts := startServer(t)
	base := ts.URL

	do(t, http.MethodPost, base+"/collections", map[string]any{
		"name": "vecs", "dim": 2, "metric": "euclidean",
	}).Body.Close()

	for i := 0; i < 5; i++ {
		do(t, http.MethodPost, base+"/collections/vecs/vectors", map[string]any{
			"id": fmt.Sprintf("v%d", i), "embeddings": []float32{float32(i), 0},
		}).Body.Close()
	}

	// No top_k — should default to 10
	resp := do(t, http.MethodPost, base+"/collections/vecs/search", map[string]any{
		"embeddings": []float32{0, 0},
	})
	var searchResp map[string][]any
	decode(t, resp, &searchResp)
	if len(searchResp["results"]) != 5 {
		t.Errorf("expected 5 results with default top_k, got %d", len(searchResp["results"]))
	}
}
