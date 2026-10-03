package index

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/fayezzouari/goatdb/core"
)

const benchDim = 128

func randomVector(dim int) core.Vector {
	emb := make([]float32, dim)
	for i := range emb {
		emb[i] = rand.Float32()*2 - 1
	}
	return core.Vector{Embeddings: emb}
}

func populateFlat(n int) *FlatIndex {
	idx := NewFlatIndex(benchDim, core.Euclidean)
	for i := 0; i < n; i++ {
		idx.AddVector(randomID(i), randomVector(benchDim))
	}
	return idx
}

func populateLSH(n int) *LSHIndex {
	idx := NewLSHIndex(benchDim, 10, 8, core.Euclidean)
	for i := 0; i < n; i++ {
		idx.AddVector(randomID(i), randomVector(benchDim))
	}
	return idx
}

func populateHNSW(n int) *HNSWIndex {
	idx := NewHNSWIndex(benchDim, 16, 200, 50, core.Euclidean)
	for i := 0; i < n; i++ {
		idx.AddVector(randomID(i), randomVector(benchDim))
	}
	return idx
}

func randomID(i int) string {
	const chars = "abcdefghijklmnopqrstuvwxyz0123456789"
	b := make([]byte, 8)
	for j := range b {
		b[j] = chars[(i+j*7)%len(chars)]
	}
	return string(b)
}

// ── AddVector ─────────────────────────────────────────────────────────────────

func BenchmarkFlatAddVector(b *testing.B) {
	idx := NewFlatIndex(benchDim, core.Euclidean)
	v := randomVector(benchDim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.AddVector(randomID(i), v)
	}
}

func BenchmarkLSHAddVector(b *testing.B) {
	idx := NewLSHIndex(benchDim, 10, 8, core.Euclidean)
	v := randomVector(benchDim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.AddVector(randomID(i), v)
	}
}

func BenchmarkHNSWAddVector(b *testing.B) {
	idx := NewHNSWIndex(benchDim, 16, 200, 50, core.Euclidean)
	v := randomVector(benchDim)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		idx.AddVector(randomID(i), v)
	}
}

// ── Search ────────────────────────────────────────────────────────────────────

var searchSizes = []int{100, 1000, 10000}

func BenchmarkFlatSearch(b *testing.B) {
	for _, n := range searchSizes {
		n := n
		idx := populateFlat(n)
		query := randomVector(benchDim)
		b.Run(sizeLabel(n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(query, 10)
			}
		})
	}
}

func BenchmarkLSHSearch(b *testing.B) {
	for _, n := range searchSizes {
		n := n
		idx := populateLSH(n)
		query := randomVector(benchDim)
		b.Run(sizeLabel(n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(query, 10)
			}
		})
	}
}

func BenchmarkHNSWSearch(b *testing.B) {
	for _, n := range searchSizes {
		n := n
		idx := populateHNSW(n)
		query := randomVector(benchDim)
		b.Run(sizeLabel(n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(query, 10)
			}
		})
	}
}

func BenchmarkIVFSearch(b *testing.B) {
	sizes := []int{100, 1000}
	for _, n := range sizes {
		n := n
		idx := NewIVFIndex(benchDim, 10, 3, core.Euclidean)
		vecs := make([]core.Vector, n)
		for i := range vecs {
			vecs[i] = randomVector(benchDim)
		}
		idx.Train(vecs)
		for i, v := range vecs {
			idx.AddVector(randomID(i), v)
		}
		query := randomVector(benchDim)
		b.Run(sizeLabel(n), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(query, 10)
			}
		})
	}
}

// ── dimensions ────────────────────────────────────────────────────────────────

var dimSizes = []int{64, 128, 512, 1536}

func BenchmarkFlatSearchByDim(b *testing.B) {
	for _, dim := range dimSizes {
		dim := dim
		idx := NewFlatIndex(dim, core.Euclidean)
		for i := 0; i < 1000; i++ {
			emb := make([]float32, dim)
			for j := range emb {
				emb[j] = rand.Float32()
			}
			idx.AddVector(randomID(i), core.Vector{Embeddings: emb})
		}
		query := make([]float32, dim)
		for j := range query {
			query[j] = rand.Float32()
		}
		b.Run(sizeLabel(dim)+"d", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(core.Vector{Embeddings: query}, 10)
			}
		})
	}
}

func BenchmarkHNSWSearchByDim(b *testing.B) {
	for _, dim := range dimSizes {
		dim := dim
		idx := NewHNSWIndex(dim, 16, 200, 50, core.Euclidean)
		for i := 0; i < 1000; i++ {
			emb := make([]float32, dim)
			for j := range emb {
				emb[j] = rand.Float32()
			}
			idx.AddVector(randomID(i), core.Vector{Embeddings: emb})
		}
		query := make([]float32, dim)
		for j := range query {
			query[j] = rand.Float32()
		}
		b.Run(sizeLabel(dim)+"d", func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.Search(core.Vector{Embeddings: query}, 10)
			}
		})
	}
}

func sizeLabel(n int) string {
	if n >= 1000 {
		return fmt.Sprintf("%dk", n/1000)
	}
	return fmt.Sprintf("%d", n)
}
