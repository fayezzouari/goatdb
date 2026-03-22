package handlers

import (
	"net/http"

	"github.com/fayez/goatdb/core"
)

type createCollectionReq struct {
	Name      string              `json:"name"`
	Dim       int                 `json:"dim"`
	Metric    core.DistanceMetric `json:"metric"`
	IndexType string              `json:"index_type"`
}

func (h *Handler) CreateCollection(w http.ResponseWriter, r *http.Request) {
	var req createCollectionReq
	if err := decode(r, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid request body")
		return
	}
	if req.Name == "" || req.Dim <= 0 {
		writeError(w, http.StatusBadRequest, "name and dim are required")
		return
	}
	if req.IndexType == "" {
		req.IndexType = "flat"
	}

	if _, err := h.DB.CreateCollection(req.Name, req.Dim, req.Metric, req.IndexType); err != nil {
		writeError(w, http.StatusConflict, err.Error())
		return
	}
	writeJSON(w, http.StatusCreated, map[string]string{"name": req.Name})
}

func (h *Handler) ListCollections(w http.ResponseWriter, r *http.Request) {
	names := h.DB.ListCollections()
	writeJSON(w, http.StatusOK, map[string][]string{"collections": names})
}

func (h *Handler) DropCollection(w http.ResponseWriter, r *http.Request) {
	if err := h.DB.DropCollection(r.PathValue("name")); err != nil {
		writeError(w, http.StatusNotFound, err.Error())
		return
	}
	w.WriteHeader(http.StatusNoContent)
}
