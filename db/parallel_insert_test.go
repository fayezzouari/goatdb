package db

import (
	"fmt"
	"math/rand"
	"sync"
	"testing"

	"github.com/fayezzouari/goatdb/core"
)

// TestAddVectorsParallelIndex inserts batches into every index type while
// searches, deletes and training run, then checks that every stored vector
// is indexed and every deleted one is gone. Run it with -race.
func TestAddVectorsParallelIndex(t *testing.T) {
	const dim, batches, batchSize = 8, 4, 300
	for _, typ := range []string{"hnsw", "flat", "ivf", "lsh"} {
		t.Run(typ, func(t *testing.T) {
			db, err := Open(t.TempDir())
			if err != nil {
				t.Fatal(err)
			}
			t.Cleanup(func() { db.Close() })
			col, err := db.CreateCollection("c", dim, core.Euclidean, typ)
			if err != nil {
				t.Fatal(err)
			}
			r := rand.New(rand.NewSource(1))
			vec := func() core.Vector {
				emb := make([]float32, dim)
				for i := range emb {
					emb[i] = r.Float32()
				}
				return core.Vector{Embeddings: emb}
			}
			all := make(map[string]core.Vector)
			var data [batches][]VectorEntry
			for b := range data {
				for i := 0; i < batchSize; i++ {
					id := fmt.Sprintf("b%d-%d", b, i)
					v := vec()
					all[id] = v
					data[b] = append(data[b], VectorEntry{Id: id, Vector: v})
				}
			}

			stop := make(chan struct{})
			var readers sync.WaitGroup
			for g := 0; g < 3; g++ {
				readers.Add(1)
				go func(g int) {
					defer readers.Done()
					q := core.Vector{Embeddings: make([]float32, dim)}
					for {
						select {
						case <-stop:
							return
						default:
						}
						if _, err := col.SearchWithOptions(ctx, q, 5, SearchOptions{}); err != nil {
							t.Error(err)
							return
						}
					}
				}(g)
			}
			for b := range data {
				if err := col.AddVectors(ctx, data[b]); err != nil {
					t.Fatal(err)
				}
				if b == 1 {
					if err := col.Train(ctx); err != nil {
						t.Fatal(err)
					}
				}
				// Delete a few vectors of this batch.
				for i := 0; i < batchSize; i += 50 {
					id := fmt.Sprintf("b%d-%d", b, i)
					if err := col.DeleteVector(ctx, id); err != nil {
						t.Fatal(err)
					}
					delete(all, id)
				}
			}
			close(stop)
			readers.Wait()

			for id, v := range all {
				if _, ok := col.index.GetVector(id); !ok {
					t.Fatalf("%s missing from the index", id)
				}
				if typ == "hnsw" || typ == "flat" {
					res, err := col.SearchWithOptions(ctx, v, 1, SearchOptions{Ef: 64})
					if err != nil {
						t.Fatal(err)
					}
					if len(res) != 1 || res[0].Id != id {
						t.Fatalf("%s: nearest is %v", id, res)
					}
				}
			}
			for b := range data {
				for i := 0; i < batchSize; i += 50 {
					if _, ok := col.index.GetVector(fmt.Sprintf("b%d-%d", b, i)); ok {
						t.Fatalf("deleted b%d-%d still indexed", b, i)
					}
				}
			}
		})
	}
}
