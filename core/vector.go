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
	if hasAVX2 && len(a) > 0 {
		return dotProductAVX2(&a[0], &b[0], len(a))
	}
	return dotScalar(a, b)
}

// L2SqSlices returns the squared euclidean distance between a and b without
// taking the square root — safe for comparison/ranking. Uses AVX2 when available.
func L2SqSlices(a, b []float32) float32 {
	if hasAVX2 && len(a) > 0 {
		return l2SquaredAVX2(&a[0], &b[0], len(a))
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

func dotScalar(a, b []float32) float32 {
	n := len(a)
	var s0, s1, s2, s3 float32
	i := 0
	for ; i <= n-8; i += 8 {
		s0 += a[i+0]*b[i+0] + a[i+1]*b[i+1]
		s1 += a[i+2]*b[i+2] + a[i+3]*b[i+3]
		s2 += a[i+4]*b[i+4] + a[i+5]*b[i+5]
		s3 += a[i+6]*b[i+6] + a[i+7]*b[i+7]
	}
	for ; i < n; i++ {
		s0 += a[i] * b[i]
	}
	return s0 + s1 + s2 + s3
}

func l2SqScalar(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}
