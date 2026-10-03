package db

import (
	"context"
	"testing"

	"github.com/fayez/goatdb/core"
	"github.com/fayez/goatdb/index"
)

// ── Search result hydration (top_k=100, dim=768) ──────────────────────────────

func BenchmarkCollectionSearchHydration(b *testing.B) {
	const dim, n, k = 768, 1000, 100
	col, err := newCollection("hydrate", dim, core.Euclidean, "hnsw", b.TempDir(), index.NewHNSWIndex(dim, 16, 200, 100, core.Euclidean))
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { col.Close() })
	ctx := context.Background()
	for i := 0; i < n; i++ {
		col.AddVector(ctx, benchID(i), core.Vector{
			Embeddings: randomEmbeddings(dim),
			Metadata:   map[string]any{"title": benchID(i), "rank": i},
		})
	}
	query := core.Vector{Embeddings: randomEmbeddings(dim)}

	cases := []struct {
		name string
		opts SearchOptions
	}{
		{"with_vectors", SearchOptions{IncludeVectors: true, IncludeMetadata: true}},
		{"without_vectors", SearchOptions{IncludeMetadata: true}},
		{"ids_only", SearchOptions{}},
	}
	// Baseline: the previous hydration path, one store.Get per result.
	b.Run("per_result_get", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			c := col
			c.mu.RLock()
			res := c.index.Search(query, k)
			for j, r := range res {
				emb, meta, err := c.store.Get(r.Id)
				if err != nil {
					continue
				}
				res[j].Vector = core.Vector{Embeddings: emb, Metadata: meta}
			}
			c.mu.RUnlock()
			if len(res) != k {
				b.Fatalf("search: %d results", len(res))
			}
		}
	})

	for _, tc := range cases {
		tc := tc
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				res, err := col.SearchWithOptions(ctx, query, k, tc.opts)
				if err != nil || len(res) != k {
					b.Fatalf("search: %d results, err=%v", len(res), err)
				}
			}
		})
	}
}
