package api

import (
	"context"
	"net/http"

	"github.com/fayez/goatdb/api/handlers"
	"github.com/fayez/goatdb/api/middleware"
	"github.com/fayez/goatdb/db"
)

type Server struct {
	httpSv *http.Server
}

func NewServer(database *db.Database, addr string) *Server {
	h := handlers.New(database)

	mux := http.NewServeMux()
	mux.HandleFunc("POST /collections", h.CreateCollection)
	mux.HandleFunc("GET /collections", h.ListCollections)
	mux.HandleFunc("DELETE /collections/{name}", h.DropCollection)
	mux.HandleFunc("POST /collections/{name}/vectors", h.AddVector)
	mux.HandleFunc("GET /collections/{name}/vectors/{id}", h.GetVector)
	mux.HandleFunc("PUT /collections/{name}/vectors/{id}", h.UpdateVector)
	mux.HandleFunc("DELETE /collections/{name}/vectors/{id}", h.DeleteVector)
	mux.HandleFunc("POST /collections/{name}/search", h.Search)
	mux.HandleFunc("POST /collections/{name}/train", h.Train)

	handler := middleware.Logging(middleware.Recovery(mux))
	return &Server{httpSv: &http.Server{Addr: addr, Handler: handler}}
}

func (s *Server) Start() error {
	return s.httpSv.ListenAndServe()
}

func (s *Server) Shutdown(ctx context.Context) error {
	return s.httpSv.Shutdown(ctx)
}
