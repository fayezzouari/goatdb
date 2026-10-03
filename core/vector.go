package core

import "math"

type Vector struct {
	Embeddings []float32
	Metadata   map[string]any
}

func (v *Vector) cosine(other *Vector) float32 {
	return cosineSlices(v.Embeddings, other.Embeddings)
}

func (v *Vector) euclidean(other *Vector) float32 {
	return float32(math.Sqrt(float64(L2SqSlices(v.Embeddings, other.Embeddings))))
}

func (v *Vector) dotProduct(other *Vector) float32 {
	return Dot(v.Embeddings, other.Embeddings)
}

func (v *Vector) manhattan(other *Vector) float32 {
	return manhattanSlices(v.Embeddings, other.Embeddings)
}

func (v *Vector) Distance(other *Vector, metric DistanceMetric) float32 {
	return DistSlices(v.Embeddings, other.Embeddings, metric)
}

// DistSlices returns the distance between a and b for metric. Indexes should
// use ResolveMetric instead, which avoids per-call dispatch and lets them skip
// cosine norms and the euclidean sqrt.
func DistSlices(a, b []float32, metric DistanceMetric) float32 {
	switch metric {
	case Cosine:
		return cosineSlices(a, b)
	case Euclidean:
		return float32(math.Sqrt(float64(L2SqSlices(a, b))))
	case DotProduct:
		return -Dot(a, b)
	case Manhattan:
		return manhattanSlices(a, b)
	default:
		panic("unsupported distance metric")
	}
}

// Dot returns the raw dot product of a and b.
func Dot(a, b []float32) float32 {
	if hasSIMD && len(a) > 0 {
		return dotSIMD(&a[0], &b[0], len(a))
	}
	return dotScalar(a, b)
}

// L2SqSlices returns the squared euclidean distance between a and b without
// taking the square root — safe for comparison/ranking. Uses SIMD when available.
func L2SqSlices(a, b []float32) float32 {
	if hasSIMD && len(a) > 0 {
		return l2SqSIMD(&a[0], &b[0], len(a))
	}
	return l2SqScalar(a, b)
}

func cosineSlices(a, b []float32) float32 {
	normA := Dot(a, a)
	normB := Dot(b, b)
	if normA == 0 || normB == 0 {
		return 0
	}
	dot := Dot(a, b)
	return 1 - (dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB)))))
}

func manhattanSlices(a, b []float32) float32 {
	var distance float32
	for i := range a {
		distance += float32(math.Abs(float64(a[i] - b[i])))
	}
	return distance
}

// dotScalar and l2SqScalar use four independent accumulators so the adds
// pipeline, and reslice both inputs so the compiler drops bounds checks.
func dotScalar(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	for len(a) >= 8 && len(b) >= 8 {
		s0 += a[0]*b[0] + a[4]*b[4]
		s1 += a[1]*b[1] + a[5]*b[5]
		s2 += a[2]*b[2] + a[6]*b[6]
		s3 += a[3]*b[3] + a[7]*b[7]
		a, b = a[8:], b[8:]
	}
	for i := range a {
		s0 += a[i] * b[i]
	}
	return (s0 + s1) + (s2 + s3)
}

func l2SqScalar(a, b []float32) float32 {
	b = b[:len(a)]
	var s0, s1, s2, s3 float32
	for len(a) >= 8 && len(b) >= 8 {
		d0, d1, d2, d3 := a[0]-b[0], a[1]-b[1], a[2]-b[2], a[3]-b[3]
		d4, d5, d6, d7 := a[4]-b[4], a[5]-b[5], a[6]-b[6], a[7]-b[7]
		s0 += d0*d0 + d4*d4
		s1 += d1*d1 + d5*d5
		s2 += d2*d2 + d6*d6
		s3 += d3*d3 + d7*d7
		a, b = a[8:], b[8:]
	}
	for i := range a {
		d := a[i] - b[i]
		s0 += d * d
	}
	return (s0 + s1) + (s2 + s3)
}
