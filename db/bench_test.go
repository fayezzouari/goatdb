package db

import (
	"context"
	"fmt"
	"math/rand"
	"testing"

	"github.com/fayezzouari/goatdb/core"
	"github.com/fayezzouari/goatdb/index"
)

const benchDim = 128

func randomEmbeddings(dim int) []float32 {
	emb := make([]float32, dim)
	for i := range emb {
		emb[i] = rand.Float32()*2 - 1
	}
	return emb
}

func benchID(i int) string {
	return fmt.Sprintf("vec-%08d", i)
}

func newBenchCollection(b *testing.B, indexType string) *Collection {
	b.Helper()
	var idx core.Index
	switch indexType {
	case "flat":
		idx = index.NewFlatIndex(benchDim, core.Euclidean)
	case "lsh":
		idx = index.NewLSHIndex(benchDim, 10, 8, core.Euclidean)
	case "hnsw":
		idx = index.NewHNSWIndex(benchDim, 16, 200, 50, core.Euclidean)
	}
	col, err := newCollection("bench", benchDim, core.Euclidean, indexType, b.TempDir(), idx)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { col.Close() })
	return col
}

// ── AddVector ─────────────────────────────────────────────────────────────────

func BenchmarkCollectionAddVector(b *testing.B) {
	for _, indexType := range []string{"flat", "lsh", "hnsw"} {
		indexType := indexType
		col := newBenchCollection(b, indexType)
		v := core.Vector{Embeddings: randomEmbeddings(benchDim)}
		next := 0
		b.Run(indexType, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := col.AddVector(context.Background(), benchID(next), v); err != nil {
					b.Fatal(err)
				}
				next++
			}
		})
	}
}

// ── AddVectors ────────────────────────────────────────────────────────────────

func BenchmarkCollectionAddVectors(b *testing.B) {
	const batchSize = 1000
	for _, indexType := range []string{"flat", "hnsw"} {
		indexType := indexType
		col := newBenchCollection(b, indexType)
		emb := randomEmbeddings(benchDim)
		next := 0
		b.Run(fmt.Sprintf("%s/batch%d", indexType, batchSize), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				batch := make([]VectorEntry, batchSize)
				for j := range batch {
					batch[j] = VectorEntry{Id: benchID(next), Vector: core.Vector{Embeddings: emb}}
					next++
				}
				if err := col.AddVectors(context.Background(), batch); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(b.N*batchSize)/b.Elapsed().Seconds(), "vectors/s")
		})
	}
}

// ── Search ────────────────────────────────────────────────────────────────────

func BenchmarkCollectionSearch(b *testing.B) {
	sizes := []int{100, 1000, 10000}
	indexTypes := []string{"flat", "lsh", "hnsw"}

	for _, indexType := range indexTypes {
		for _, n := range sizes {
			indexType, n := indexType, n
			col := newBenchCollection(b, indexType)
			ctx := context.Background()
			for i := 0; i < n; i++ {
				col.AddVector(ctx, benchID(i), core.Vector{Embeddings: randomEmbeddings(benchDim)})
			}
			query := core.Vector{Embeddings: randomEmbeddings(benchDim)}
			b.Run(fmt.Sprintf("%s/n%d", indexType, n), func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					col.Search(ctx, query, 10)
				}
			})
		}
	}
}

// ── GetVector ─────────────────────────────────────────────────────────────────

func BenchmarkCollectionGetVector(b *testing.B) {
	col := newBenchCollection(b, "flat")
	ctx := context.Background()
	for i := 0; i < 1000; i++ {
		col.AddVector(ctx, benchID(i), core.Vector{Embeddings: randomEmbeddings(benchDim)})
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		col.GetVector(ctx, benchID(i%1000))
	}
}

// ── Parallel search ───────────────────────────────────────────────────────────

func BenchmarkCollectionSearchParallel(b *testing.B) {
	for _, indexType := range []string{"flat", "hnsw"} {
		indexType := indexType
		col := newBenchCollection(b, indexType)
		ctx := context.Background()
		for i := 0; i < 1000; i++ {
			col.AddVector(ctx, benchID(i), core.Vector{Embeddings: randomEmbeddings(benchDim)})
		}
		query := core.Vector{Embeddings: randomEmbeddings(benchDim)}
		b.Run(indexType, func(b *testing.B) {
			b.ResetTimer()
			b.RunParallel(func(pb *testing.PB) {
				for pb.Next() {
					col.Search(ctx, query, 10)
				}
			})
		})
	}
}

// ── top_k sensitivity ─────────────────────────────────────────────────────────

func BenchmarkCollectionSearchTopK(b *testing.B) {
	col := newBenchCollection(b, "hnsw")
	ctx := context.Background()
	for i := 0; i < 5000; i++ {
		col.AddVector(ctx, benchID(i), core.Vector{Embeddings: randomEmbeddings(benchDim)})
	}
	query := core.Vector{Embeddings: randomEmbeddings(benchDim)}
	for _, k := range []int{1, 10, 50, 100} {
		k := k
		b.Run(fmt.Sprintf("k%d", k), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				col.Search(ctx, query, k)
			}
		})
	}
}
