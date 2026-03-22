package handlers

import (
	"net/http"

	"github.com/fayez/goatdb/core"
)

type vectorReq struct {
	Embeddings []float32      `json:"embeddings"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

func (h *Handler) AddVector(w http.ResponseWriter, r *http.Request) {
	col, err := h.DB.GetCollection(r.PathValue("name"))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	var req struct {
		Id string `json:"id"`
		vectorReq
	}
	if err := decode(r, &req); err != nil || req.Id == "" || len(req.Embeddings) == 0 {
		writeError(w, http.StatusBadRequest, "id and embeddings are required")
		return
	}

	if err := col.AddVector(r.Context(), req.Id, core.Vector{Embeddings: req.Embeddings, Metadata: req.Metadata}); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"id": req.Id})
}

func (h *Handler) AddVectors(w http.ResponseWriter, r *http.Request) {
	col, err := h.DB.GetCollection(r.PathValue("name"))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	var req struct {
		Vectors []struct {
			Id string `json:"id"`
			vectorReq
		} `json:"vectors"`
	}
	if err := decode(r, &req); err != nil || len(req.Vectors) == 0 {
		writeError(w, http.StatusBadRequest, "vectors array is required")
		return
	}

	batch := make(map[string]core.Vector, len(req.Vectors))
	for _, v := range req.Vectors {
		if v.Id == "" || len(v.Embeddings) == 0 {
			writeError(w, http.StatusBadRequest, "each vector requires id and embeddings")
			return
		}
		batch[v.Id] = core.Vector{Embeddings: v.Embeddings, Metadata: v.Metadata}
	}

	if err := col.AddVectors(r.Context(), batch); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]int{"inserted": len(batch)})
}

func (h *Handler) GetVector(w http.ResponseWriter, r *http.Request) {
	col, err := h.DB.GetCollection(r.PathValue("name"))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	id := r.PathValue("id")
	v, ok := col.GetVector(r.Context(), id)
	if !ok {
		writeError(w, http.StatusNotFound, "vector not found")
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"id":         id,
		"embeddings": v.Embeddings,
		"metadata":   v.Metadata,
	})
}

func (h *Handler) UpdateVector(w http.ResponseWriter, r *http.Request) {
	col, err := h.DB.GetCollection(r.PathValue("name"))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	var req vectorReq
	if err := decode(r, &req); err != nil || len(req.Embeddings) == 0 {
		writeError(w, http.StatusBadRequest, "embeddings are required")
		return
	}

	if err := col.UpdateVector(r.Context(), r.PathValue("id"), core.Vector{Embeddings: req.Embeddings, Metadata: req.Metadata}); err != nil {
		writeError(w, http.StatusBadRequest, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (h *Handler) DeleteVector(w http.ResponseWriter, r *http.Request) {
	col, err := h.DB.GetCollection(r.PathValue("name"))
	if err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}

	if err := col.DeleteVector(r.Context(), r.PathValue("id")); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
