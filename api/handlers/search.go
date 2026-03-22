package handlers

import (
	"net/http"

	"github.com/fayez/goatdb/core"
)

type searchReq struct {
	Embeddings []float32 `json:"embeddings"`
	TopK       int       `json:"top_k"`
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

	results, err := col.Search(core.Vector{Embeddings: req.Embeddings}, req.TopK)
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

	if err := col.Train(); err != nil {
		writeError(w, http.StatusInternalServerError, err.Error())
		return
	}
	writeJSON(w, http.StatusOK, map[string]string{"status": "trained"})
}
