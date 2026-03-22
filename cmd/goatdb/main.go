package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/fayez/goatdb/api"
	"github.com/fayez/goatdb/db"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	dir := flag.String("dir", "./data", "data directory")
	flag.Parse()

	database, err := db.Open(*dir)
	if err != nil {
		log.Fatalf("open db: %v", err)
	}

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
