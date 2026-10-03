package handlers

import (
	"errors"
	"fmt"
	"net/http"

	"github.com/fayezzouari/goatdb/core"
	"github.com/fayezzouari/goatdb/db"
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
		writeError(w, insertErrorStatus(err), err.Error())
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

	batch := make([]db.VectorEntry, 0, len(req.Vectors))
	seen := make(map[string]struct{}, len(req.Vectors))
	for _, v := range req.Vectors {
		if v.Id == "" || len(v.Embeddings) == 0 {
			writeError(w, http.StatusBadRequest, "each vector requires id and embeddings")
			return
		}
		if _, dup := seen[v.Id]; dup {
			writeError(w, http.StatusBadRequest, fmt.Sprintf("duplicate id %q in batch", v.Id))
			return
		}
		seen[v.Id] = struct{}{}
		batch = append(batch, db.VectorEntry{Id: v.Id, Vector: core.Vector{Embeddings: v.Embeddings, Metadata: v.Metadata}})
	}

	if err := col.AddVectors(r.Context(), batch); err != nil {
		writeError(w, insertErrorStatus(err), err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]int{"inserted": len(batch)})
}

func insertErrorStatus(err error) int {
	if errors.Is(err, db.ErrExists) {
		return http.StatusConflict
	}
	return http.StatusBadRequest
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
