package core

import "math"

// DistFunc computes a distance between two vectors of equal length.
type DistFunc func(a, b []float32) float32

// Metric is a DistanceMetric resolved once for use inside an index.
//
// Dist is a ranking distance, monotone with the metric's reported distance:
// for Cosine it expects inputs prepared with Prepare (L2-normalized) and for
// Euclidean it returns the squared distance. Finalize maps a Dist value back
// to the reported distance.
type Metric struct {
	Dist      DistFunc
	normalize bool
	squared   bool
}

func ResolveMetric(m DistanceMetric) Metric {
	switch m {
	case Cosine:
		return Metric{Dist: cosineNormalized, normalize: true}
	case Euclidean:
		return Metric{Dist: L2SqSlices, squared: true}
	case DotProduct:
		return Metric{Dist: negDot}
	case Manhattan:
		return Metric{Dist: manhattanSlices}
	default:
		return Metric{Dist: func(a, b []float32) float32 { return DistSlices(a, b, m) }}
	}
}

// Normalizes reports whether vectors must be L2-normalized before Dist.
func (m Metric) Normalizes() bool { return m.normalize }

// Prepare returns v in the form Dist expects. It never modifies v.
func (m Metric) Prepare(v []float32) []float32 {
	if !m.normalize {
		return v
	}
	out := make([]float32, len(v))
	copy(out, v)
	Normalize(out)
	return out
}

// PrepareInPlace is Prepare for a slice the caller owns.
func (m Metric) PrepareInPlace(v []float32) {
	if m.normalize {
		Normalize(v)
	}
}

// Finalize converts a Dist value into the metric's reported distance.
func (m Metric) Finalize(d float32) float32 {
	if m.squared {
		return float32(math.Sqrt(float64(d)))
	}
	return d
}

// Normalize scales v to unit L2 norm in place. Zero vectors are left unchanged.
func Normalize(v []float32) {
	n := Dot(v, v)
	if n == 0 {
		return
	}
	inv := float32(1 / math.Sqrt(float64(n)))
	for i := range v {
		v[i] *= inv
	}
}

func cosineNormalized(a, b []float32) float32 { return 1 - Dot(a, b) }

func negDot(a, b []float32) float32 { return -Dot(a, b) }
