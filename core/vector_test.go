package core

import (
	"math"
	"testing"
)

func TestDotProduct(t *testing.T) {
	a := &Vector{Embeddings: []float32{1, 2, 3}}
	b := &Vector{Embeddings: []float32{4, 5, 6}}
	got := a.dotProduct(b)
	want := float32(32) // 1*4 + 2*5 + 3*6
	if got != want {
		t.Errorf("dotProduct = %v, want %v", got, want)
	}
}

func TestDotProductUnrolled(t *testing.T) {
	n := 16
	a := &Vector{Embeddings: make([]float32, n)}
	b := &Vector{Embeddings: make([]float32, n)}
	var want float32
	for i := 0; i < n; i++ {
		a.Embeddings[i] = float32(i + 1)
		b.Embeddings[i] = float32(i + 1)
		want += float32((i + 1) * (i + 1))
	}
	got := a.dotProduct(b)
	if math.Abs(float64(got-want)) > 1e-2 {
		t.Errorf("dotProduct unrolled = %v, want %v", got, want)
	}
}

func TestCosineIdentical(t *testing.T) {
	a := &Vector{Embeddings: []float32{1, 0, 0}}
	got := a.cosine(a)
	if math.Abs(float64(got)) > 1e-6 {
		t.Errorf("cosine of identical vectors = %v, want 0", got)
	}
}

func TestCosineOrthogonal(t *testing.T) {
	a := &Vector{Embeddings: []float32{1, 0}}
	b := &Vector{Embeddings: []float32{0, 1}}
	got := a.cosine(b)
	want := float32(1) // distance = 1 - cos(90°) = 1 - 0 = 1
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("cosine of orthogonal vectors = %v, want %v", got, want)
	}
}

func TestEuclidean(t *testing.T) {
	a := &Vector{Embeddings: []float32{0, 0}}
	b := &Vector{Embeddings: []float32{3, 4}}
	got := a.euclidean(b)
	want := float32(5)
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Errorf("euclidean = %v, want %v", got, want)
	}
}

func TestManhattan(t *testing.T) {
	a := &Vector{Embeddings: []float32{1, 2, 3}}
	b := &Vector{Embeddings: []float32{4, 5, 6}}
	got := a.manhattan(b)
	want := float32(9) // |1-4| + |2-5| + |3-6|
	if got != want {
		t.Errorf("manhattan = %v, want %v", got, want)
	}
}

func TestDistanceDispatch(t *testing.T) {
	a := &Vector{Embeddings: []float32{1, 0}}
	b := &Vector{Embeddings: []float32{1, 0}}

	metrics := []DistanceMetric{Cosine, Euclidean, DotProduct, Manhattan}
	for _, m := range metrics {
		got := a.Distance(b, m)
		if math.IsNaN(float64(got)) {
			t.Errorf("Distance(%v) returned NaN", m)
		}
	}
}
