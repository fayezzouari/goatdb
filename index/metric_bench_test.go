package index

import (
	"strconv"
	"testing"

	"github.com/fayezzouari/goatdb/core"
)

var benchMetrics = []core.DistanceMetric{core.Cosine, core.Euclidean, core.DotProduct}

func BenchmarkFlatSearchByMetric(b *testing.B) {
	for _, m := range benchMetrics {
		idx := NewFlatIndex(benchDim, m)
		for i := 0; i < 10000; i++ {
			idx.AddVector(strconv.Itoa(i), randomVector(benchDim))
		}
		query := randomVector(benchDim)
		b.Run(string(m), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				idx.Search(query, 10)
			}
		})
	}
}

func BenchmarkHNSWAddVectorByMetric(b *testing.B) {
	for _, m := range benchMetrics {
		b.Run(string(m), func(b *testing.B) {
			idx := NewHNSWIndex(benchDim, 16, 200, 50, m)
			vecs := make([]core.Vector, 1024)
			for i := range vecs {
				vecs[i] = randomVector(benchDim)
			}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				idx.AddVector(strconv.Itoa(i), vecs[i%len(vecs)])
			}
		})
	}
}

func BenchmarkHNSWSearchByMetric(b *testing.B) {
	for _, m := range benchMetrics {
		idx := NewHNSWIndex(benchDim, 16, 200, 50, m)
		for i := 0; i < 10000; i++ {
			idx.AddVector(strconv.Itoa(i), randomVector(benchDim))
		}
		query := randomVector(benchDim)
		b.Run(string(m), func(b *testing.B) {
			for i := 0; i < b.N; i++ {
				idx.Search(query, 10)
			}
		})
	}
}
