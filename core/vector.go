package core

import "math"

type Vector struct {
	Embeddings []float32
	Metadata   map[string]any
}

func (v *Vector) cosine(other *Vector) float32 {
	normV1 := float32(0)
	normV2 := float32(0)
	n := len(v.Embeddings)
	for i := 0; i < n; i++ {
		normV1 += v.Embeddings[i] * v.Embeddings[i]
		normV2 += other.Embeddings[i] * other.Embeddings[i]
	}
	dot := v.dotProduct(other)
	if normV1 == 0 || normV2 == 0 {
		return 0
	}
	return 1 - (dot / (float32(math.Sqrt(float64(normV1))) * float32(math.Sqrt(float64(normV2)))))
}

func (v *Vector) euclidean(other *Vector) float32 {
	distance := float32(0)
	n := len(v.Embeddings)
	for i := 0; i < n; i++ {
		diff := v.Embeddings[i] - other.Embeddings[i]
		distance += diff * diff
	}
	return float32(math.Sqrt(float64(distance)))
}

func (v *Vector) dotProduct(other *Vector) float32 {
	if len(v.Embeddings) != len(other.Embeddings) {
		panic("Vectors must be of the same length")
	}
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

func (v *Vector) manhattan(other *Vector) float32 {
	distance := float32(0)
	n := len(v.Embeddings)
	for i := 0; i < n; i++ {
		distance += float32(math.Abs(float64(v.Embeddings[i] - other.Embeddings[i])))
	}
	return distance
}
