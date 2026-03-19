package core

import "math"

type Vector struct {
	id         int
	embeddings []float32
	metadata   map[string]any
}

func cosine(v1 []float32, v2 []float32) float32 {
}

func euclidean(v1 []float32, v2 []float32) float32 {
	distance := float32(0)
	n := len(v1)
	for i := 0; i < n; i++ {
		diff := v1[i] - v2[i]
		distance += diff * diff
	}
	return float32(math.Sqrt(float64(distance)))
}

func dotProduct(v1 []float32, v2 []float32) float32 {
	if len(v1) != len(v2) {
		panic("Vectors must be of the same length")
	}
	n := len(v1)
	var s0, s1, s2, s3 float32
	i := 0
	for ; i <= n-8; i += 8 {
		s0 += v1[i+0]*v2[i+0] + v1[i+1]*v2[i+1]
		s1 += v1[i+2]*v2[i+2] + v1[i+3]*v2[i+3]
		s2 += v1[i+4]*v2[i+4] + v1[i+5]*v2[i+5]
		s3 += v1[i+6]*v2[i+6] + v1[i+7]*v2[i+7]
	}
	for ; i < n; i++ {
		s0 += v1[i] * v2[i]
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
