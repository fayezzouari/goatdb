package core

import "math"

type Vector struct {
	Embeddings []float32
	Metadata   map[string]any
}

func (v *Vector) cosine(other *Vector) float32 {
	var normV1, normV2, dot float32
	if hasAVX2 && len(v.Embeddings) > 0 {
		normV1 = dotProductAVX2(&v.Embeddings[0], &v.Embeddings[0], len(v.Embeddings))
		normV2 = dotProductAVX2(&other.Embeddings[0], &other.Embeddings[0], len(other.Embeddings))
		dot = dotProductAVX2(&v.Embeddings[0], &other.Embeddings[0], len(v.Embeddings))
	} else {
		n := len(v.Embeddings)
		for i := 0; i < n; i++ {
			normV1 += v.Embeddings[i] * v.Embeddings[i]
			normV2 += other.Embeddings[i] * other.Embeddings[i]
		}
		dot = v.dotProductScalar(other)
	}
	if normV1 == 0 || normV2 == 0 {
		return 0
	}
	return 1 - (dot / (float32(math.Sqrt(float64(normV1))) * float32(math.Sqrt(float64(normV2)))))
}

func (v *Vector) euclidean(other *Vector) float32 {
	if hasAVX2 && len(v.Embeddings) > 0 {
		return float32(math.Sqrt(float64(l2SquaredAVX2(&v.Embeddings[0], &other.Embeddings[0], len(v.Embeddings)))))
	}
	distance := float32(0)
	n := len(v.Embeddings)
	for i := 0; i < n; i++ {
		diff := v.Embeddings[i] - other.Embeddings[i]
		distance += diff * diff
	}
	return float32(math.Sqrt(float64(distance)))
}

func (v *Vector) dotProduct(other *Vector) float32 {
	if hasAVX2 && len(v.Embeddings) > 0 {
		return dotProductAVX2(&v.Embeddings[0], &other.Embeddings[0], len(v.Embeddings))
	}
	return v.dotProductScalar(other)
}

func (v *Vector) dotProductScalar(other *Vector) float32 {
	n := len(v.Embeddings)
	var s0, s1, s2, s3 float32
	i := 0
	for ; i <= n-8; i += 8 {
		s0 += v.Embeddings[i+0]*other.Embeddings[i+0] + v.Embeddings[i+1]*other.Embeddings[i+1]
		s1 += v.Embeddings[i+2]*other.Embeddings[i+2] + v.Embeddings[i+3]*other.Embeddings[i+3]
		s2 += v.Embeddings[i+4]*other.Embeddings[i+4] + v.Embeddings[i+5]*other.Embeddings[i+5]
		s3 += v.Embeddings[i+6]*other.Embeddings[i+6] + v.Embeddings[i+7]*other.Embeddings[i+7]
	}
	for ; i < n; i++ {
		s0 += v.Embeddings[i] * other.Embeddings[i]
	}
	return s0 + s1 + s2 + s3
}

func (v *Vector) Distance(other *Vector, metric DistanceMetric) float32 {
	switch metric {
	case Cosine:
		return v.cosine(other)
	case Euclidean:
		return v.euclidean(other)
	case DotProduct:
		return v.dotProduct(other)
	case Manhattan:
		return v.manhattan(other)
	default:
		panic("unsupported distance metric")
	}
}

func DistSlices(a, b []float32, metric DistanceMetric) float32 {
	va := Vector{Embeddings: a}
	vb := Vector{Embeddings: b}
	return va.Distance(&vb, metric)
}

// L2SqSlices returns the squared euclidean distance between a and b without
// taking the square root — safe for comparison/ranking. Uses AVX2 when available.
func L2SqSlices(a, b []float32) float32 {
	if hasAVX2 && len(a) > 0 {
		return l2SquaredAVX2(&a[0], &b[0], len(a))
	}
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

func (v *Vector) manhattan(other *Vector) float32 {
	distance := float32(0)
	n := len(v.Embeddings)
	for i := 0; i < n; i++ {
		distance += float32(math.Abs(float64(v.Embeddings[i] - other.Embeddings[i])))
	}
	return distance
}
