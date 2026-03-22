package core

import (
	"math"
	"math/rand"
)

type SQCodebook struct {
	Min   float32
	Scale float32
	Dim   int
}

func NewSQCodebook(vectors []Vector) *SQCodebook {
	if len(vectors) == 0 {
		return nil
	}
	min := float32(math.MaxFloat32)
	max := float32(-math.MaxFloat32)
	for _, v := range vectors {
		for _, x := range v.Embeddings {
			if x < min {
				min = x
			}
			if x > max {
				max = x
			}
		}
	}
	if max == min {
		return nil
	}
	return &SQCodebook{Min: min, Scale: (max - min) / 255.0, Dim: len(vectors[0].Embeddings)}
}

func (cb *SQCodebook) Quantize(v []float32) []int8 {
	q := make([]int8, cb.Dim)
	inv := 1.0 / cb.Scale
	for i, x := range v {
		code := (x - cb.Min) * inv
		if code < 0 {
			code = 0
		} else if code > 255 {
			code = 255
		}
		q[i] = int8(int32(code) - 128)
	}
	return q
}

// DistInt8 returns an unnormalized approximate l2-squared distance in
// quantized int8 space. Monotone with the true euclidean distance, so
// safe for ranking candidates during graph traversal.
func (cb *SQCodebook) DistInt8(a, b []int8) float32 {
	var sum int32
	for i := range a {
		d := int32(a[i]) - int32(b[i])
		sum += d * d
	}
	return float32(sum)
}

// Int8VectorPool is a flat []int8 backing array indexed by the same slot
// numbers as a VectorPool. It has no slot management of its own; liveness
// is determined by the paired VectorPool.
type Int8VectorPool struct {
	data []int8
	dim  int
}

func NewInt8VectorPool(dim, slots int) *Int8VectorPool {
	return &Int8VectorPool{
		dim:  dim,
		data: make([]int8, slots*dim),
	}
}

func (p *Int8VectorPool) Set(idx int32, emb []int8) {
	start := int(idx) * p.dim
	copy(p.data[start:], emb[:p.dim])
}

func (p *Int8VectorPool) Get(idx int32) []int8 {
	start := int(idx) * p.dim
	return p.data[start : start+p.dim]
}

func (p *Int8VectorPool) Grow(n int) {
	need := n * p.dim
	for len(p.data) < need {
		p.data = append(p.data, 0)
	}
}
