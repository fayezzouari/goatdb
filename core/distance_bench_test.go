package core

import (
	"fmt"
	"math/rand"
	"testing"
)

var benchDims = []int{128, 768, 1536}

var sinkF32 float32

func benchPair(dim int) ([]float32, []float32) {
	r := rand.New(rand.NewSource(int64(dim)))
	a := make([]float32, dim)
	b := make([]float32, dim)
	for i := range a {
		a[i] = r.Float32()*2 - 1
		b[i] = r.Float32()*2 - 1
	}
	return a, b
}

func benchMetric(b *testing.B, metric DistanceMetric) {
	for _, dim := range benchDims {
		x, y := benchPair(dim)
		b.Run(fmt.Sprintf("%d", dim), func(b *testing.B) {
			b.SetBytes(int64(dim * 8))
			for i := 0; i < b.N; i++ {
				sinkF32 = DistSlices(x, y, metric)
			}
		})
	}
}

func BenchmarkDistCosine(b *testing.B)     { benchMetric(b, Cosine) }
func BenchmarkDistEuclidean(b *testing.B)  { benchMetric(b, Euclidean) }
func BenchmarkDistDotProduct(b *testing.B) { benchMetric(b, DotProduct) }

// BenchmarkRank measures the per-comparison cost inside an index: inputs are
// prepared once and Dist is the resolved ranking function.
func BenchmarkRank(b *testing.B) {
	for _, metric := range []DistanceMetric{Cosine, Euclidean, DotProduct} {
		m := ResolveMetric(metric)
		for _, dim := range benchDims {
			x, y := benchPair(dim)
			x, y = m.Prepare(x), m.Prepare(y)
			b.Run(fmt.Sprintf("%s/%d", metric, dim), func(b *testing.B) {
				b.SetBytes(int64(dim * 8))
				for i := 0; i < b.N; i++ {
					sinkF32 = m.Dist(x, y)
				}
			})
		}
	}
}

// BenchmarkScalar measures the pure-Go fallback used on CPUs without SIMD.
func BenchmarkScalar(b *testing.B) {
	for _, k := range []struct {
		name string
		fn   func(a, b []float32) float32
	}{{"dot", dotScalar}, {"l2sq", l2SqScalar}} {
		for _, dim := range benchDims {
			x, y := benchPair(dim)
			b.Run(fmt.Sprintf("%s/%d", k.name, dim), func(b *testing.B) {
				b.SetBytes(int64(dim * 8))
				for i := 0; i < b.N; i++ {
					sinkF32 = k.fn(x, y)
				}
			})
		}
	}
}

func BenchmarkL2Sq(b *testing.B) {
	for _, dim := range benchDims {
		x, y := benchPair(dim)
		b.Run(fmt.Sprintf("%d", dim), func(b *testing.B) {
			b.SetBytes(int64(dim * 8))
			for i := 0; i < b.N; i++ {
				sinkF32 = L2SqSlices(x, y)
			}
		})
	}
}
