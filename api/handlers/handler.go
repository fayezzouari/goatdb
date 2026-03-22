package handlers

import (
	"encoding/json"
	"net/http"

	"github.com/fayez/goatdb/db"
)

type Handler struct {
	DB *db.Database
}

func New(database *db.Database) *Handler {
	return &Handler{DB: database}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeError(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]string{"error": msg})
}

func decode(r *http.Request, dst any) error {
	return json.NewDecoder(r.Body).Decode(dst)
}
