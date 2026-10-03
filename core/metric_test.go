package core

import (
	"math"
	"math/rand"
	"slices"
	"testing"
)

func TestResolveMetricMatchesDistSlices(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for _, metric := range []DistanceMetric{Cosine, Euclidean, DotProduct, Manhattan} {
		m := ResolveMetric(metric)
		for _, dim := range []int{1, 3, 8, 17, 128} {
			a := make([]float32, dim)
			b := make([]float32, dim)
			for i := range a {
				a[i] = (r.Float32()*2 - 1) * 5
				b[i] = (r.Float32()*2 - 1) * 5
			}
			ac, bc := slices.Clone(a), slices.Clone(b)
			got := m.Finalize(m.Dist(m.Prepare(a), m.Prepare(b)))
			want := DistSlices(a, b, metric)
			if math.Abs(float64(got-want)) > 1e-4*math.Max(1, math.Abs(float64(want))) {
				t.Errorf("%s dim=%d: got %v, want %v", metric, dim, got, want)
			}
			if !slices.Equal(a, ac) || !slices.Equal(b, bc) {
				t.Errorf("%s: Prepare modified its input", metric)
			}
		}
	}
}

func TestNormalize(t *testing.T) {
	v := []float32{3, 4}
	Normalize(v)
	if math.Abs(float64(v[0]-0.6)) > 1e-6 || math.Abs(float64(v[1]-0.8)) > 1e-6 {
		t.Errorf("Normalize = %v, want [0.6 0.8]", v)
	}
	z := []float32{0, 0}
	Normalize(z)
	if z[0] != 0 || z[1] != 0 {
		t.Errorf("Normalize(zero) = %v", z)
	}
}
