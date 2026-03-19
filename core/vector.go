package core

import "math"

type Vector struct {
	id         int
	embeddings []float32
	metadata   map[string]any
}

func (v *Vector) cosine(other *Vector) float32 {
	normV1 := float32(0)
	normV2 := float32(0)
	n := len(v.embeddings)
	for i := 0; i < n; i++ {
		normV1 += v.embeddings[i] * v.embeddings[i]
		normV2 += other.embeddings[i] * other.embeddings[i]
	}
	dot := dotProduct(v.embeddings, other.embeddings)
	if normV1 == 0 || normV2 == 0 {
		return 0
	}
	return 1 - (dot / (float32(math.Sqrt(float64(normV1))) * float32(math.Sqrt(float64(normV2)))))
}

func (v *Vector) euclidean(other *Vector) float32 {
	distance := float32(0)
	n := len(v.embeddings)
	for i := 0; i < n; i++ {
		diff := v.embeddings[i] - other.embeddings[i]
		distance += diff * diff
	}
	return float32(math.Sqrt(float64(distance)))
}

func (v *Vector) dotProduct(other *Vector) float32 {
	if len(v.embeddings) != len(other.embeddings) {
		panic("Vectors must be of the same length")
	}
	n := len(v.embeddings)
	var s0, s1, s2, s3 float32
	i := 0
	for ; i <= n-8; i += 8 {
		s0 += v.embeddings[i+0]*other.embeddings[i+0] + v.embeddings[i+1]*other.embeddings[i+1]
		s1 += v.embeddings[i+2]*other.embeddings[i+2] + v.embeddings[i+3]*other.embeddings[i+3]
		s2 += v.embeddings[i+4]*other.embeddings[i+4] + v.embeddings[i+5]*other.embeddings[i+5]
		s3 += v.embeddings[i+6]*other.embeddings[i+6] + v.embeddings[i+7]*other.embeddings[i+7]
	}
	for ; i < n; i++ {
		s0 += v.embeddings[i] * other.embeddings[i]
	}
	return s0 + s1 + s2 + s3
}

func manhattan(v1 []float32, v2 []float32) float32 {
	distance := float32(0)
	n := len(v1)
	for i := 0; i < n; i++ {
		distance += float32(math.Abs(float64(v1[i] - v2[i])))
	}
	return distance
}
