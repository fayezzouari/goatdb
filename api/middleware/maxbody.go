package middleware

import (
	"net/http"
)

const defaultMaxBodyBytes = 32 << 20 // 32 MB

func MaxBody(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		r.Body = http.MaxBytesReader(w, r.Body, defaultMaxBodyBytes)
		next.ServeHTTP(w, r)
	})
}
