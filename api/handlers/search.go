package handlers

import (
	"net/http"

	"github.com/fayezzouari/goatdb/core"
	"github.com/fayezzouari/goatdb/db"
)

type searchReq struct {
	Embeddings []float32 `json:"embeddings"`
	TopK       int       `json:"top_k"`
	// IncludeVectors defaults to false; IncludeMetadata defaults to true.
	IncludeVectors  bool  `json:"include_vectors"`
	IncludeMetadata *bool `json:"include_metadata"`
}

func (h *Handler) Search(w http.ResponseWriter, r *http.Request) {
	col, err := h.DB.GetCollection(r.PathValue("name"))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	var req searchReq
	if err := decode(r, &req); err != nil || len(req.Embeddings) == 0 {
		writeError(w, http.StatusBadRequest, "embeddings are required")
		return
	}
	if req.TopK <= 0 {
		req.TopK = 10
	}

	opts := db.SearchOptions{IncludeVectors: req.IncludeVectors, IncludeMetadata: true}
	if req.IncludeMetadata != nil {
		opts.IncludeMetadata = *req.IncludeMetadata
	}

	results, err := col.SearchWithOptions(r.Context(), core.Vector{Embeddings: req.Embeddings}, req.TopK, opts)
	if err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}

	type resultItem struct {
		Id         string         `json:"id"`
		Distance   float32        `json:"distance"`
		Embeddings []float32      `json:"embeddings,omitempty"`
		Metadata   map[string]any `json:"metadata,omitempty"`
	}
	out := make([]resultItem, len(results))
	for i, res := range results {
		out[i] = resultItem{
			Id:         res.Id,
			Distance:   res.Distance,
			Embeddings: res.Vector.Embeddings,
			Metadata:   res.Vector.Metadata,
		}
	}
	writeJSON(w, http.StatusOK, map[string]any{"results": out})
}

func (h *Handler) Train(w http.ResponseWriter, r *http.Request) {
	col, err := h.DB.GetCollection(r.PathValue("name"))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	if err := col.Train(r.Context()); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "trained"})
}
