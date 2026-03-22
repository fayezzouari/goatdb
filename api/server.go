package api

import (
	"context"
	"encoding/json"
	"net/http"
	"time"

	"github.com/fayez/goatdb/api/handlers"
	"github.com/fayez/goatdb/api/middleware"
	"github.com/fayez/goatdb/db"
)

type Server struct {
	httpSv  *http.Server
	Metrics *middleware.Metrics
}

func NewServer(database *db.Database, addr string) *Server {
	h := handlers.New(database)
	m := &middleware.Metrics{}

	mux := http.NewServeMux()

	// Health & metrics
	mux.HandleFunc("GET /health", health)
	mux.HandleFunc("GET /metrics", m.Handler())

	// Collections
	mux.HandleFunc("POST /collections", h.CreateCollection)
	mux.HandleFunc("GET /collections", h.ListCollections)
	mux.HandleFunc("GET /collections/{name}", h.GetCollection)
	mux.HandleFunc("DELETE /collections/{name}", h.DropCollection)

	// Vectors
	mux.HandleFunc("POST /collections/{name}/vectors", h.AddVector)
	mux.HandleFunc("POST /collections/{name}/vectors/batch", h.AddVectors)
	mux.HandleFunc("GET /collections/{name}/vectors/{id}", h.GetVector)
	mux.HandleFunc("PUT /collections/{name}/vectors/{id}", h.UpdateVector)
	mux.HandleFunc("DELETE /collections/{name}/vectors/{id}", h.DeleteVector)

	// Search & training
	mux.HandleFunc("POST /collections/{name}/search", h.Search)
	mux.HandleFunc("POST /collections/{name}/train", h.Train)

	handler := m.Collect(middleware.Logging(middleware.Recovery(middleware.MaxBody(mux))))
	return &Server{
		httpSv:  &http.Server{Addr: addr, Handler: handler},
		Metrics: m,
	}
}

// Handler returns the HTTP handler for use in tests.
func (s *Server) Handler() http.Handler {
	return s.httpSv.Handler
}

func health(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]any{
		"status": "ok",
		"time":   time.Now().UTC(),
	})
}

func (s *Server) Start() error {
	return s.httpSv.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpSv.Shutdown(ctx)
}
