package middleware

import (
	"encoding/json"
	"net/http"
	"sync/atomic"
)

type Metrics struct {
	RequestsTotal atomic.Int64
	Status2xx     atomic.Int64
	Status4xx     atomic.Int64
	Status5xx     atomic.Int64
}

func (m *Metrics) Handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]int64{
			"requests_total": m.RequestsTotal.Load(),
			"status_2xx":     m.Status2xx.Load(),
			"status_4xx":     m.Status4xx.Load(),
			"status_5xx":     m.Status5xx.Load(),
		})
	}
}

func (m *Metrics) Collect(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rw := &responseWriter{ResponseWriter: w, status: http.StatusOK}
		next.ServeHTTP(rw, r)
		m.RequestsTotal.Add(1)
		switch {
		case rw.status < 400:
			m.Status2xx.Add(1)
		case rw.status < 500:
			m.Status4xx.Add(1)
		default:
			m.Status5xx.Add(1)
		}
	})
}
