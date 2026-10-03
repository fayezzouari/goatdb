package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fayez/goatdb/api"
	"github.com/fayez/goatdb/db"
)

// version is set at build time via -ldflags "-X main.version=...".
var version = "dev"

// envOr returns the value of the environment variable key, or def if it is unset or empty.
func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

func main() {
	addr := flag.String("addr", envOr("GOATDB_ADDR", ":8080"), "listen address (env GOATDB_ADDR)")
	dir := flag.String("dir", envOr("GOATDB_DIR", "./data"), "data directory (env GOATDB_DIR)")
	showVersion := flag.Bool("version", false, "print version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println("goatdb", version)
		return
	}
	log.Printf("goatdb %s", version)

	database, err := db.Open(*dir)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}
	log.Printf("data dir: %s", *dir)

	srv := api.NewServer(database, *addr)

	go func() {
		log.Printf("goatdb listening on %s", *addr)
		if err := srv.Start(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("server: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	log.Println("shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := srv.Shutdown(ctx); err != nil {
		log.Printf("shutdown: %v", err)
	}
	if err := database.Close(); err != nil {
		log.Printf("db close: %v", err)
	}
}
