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

// PQCodebook implements Product Quantization: the vector space is split into
// NSubs subspaces each of SubDim dimensions, and each subspace is quantized
// independently with NCentroids k-means centroids (typically 256 → 1 byte).
// Encoding a vector yields NSubs bytes; distances are computed asymmetrically
// using a precomputed lookup table — O(NSubs) instead of O(Dim).
type PQCodebook struct {
	NSubs      int
	NCentroids int
	SubDim     int
	Centroids  []float32 // [NSubs * NCentroids * SubDim]
}

// NewPQCodebook trains a PQ codebook from the given vectors.
// nSubs must divide dim evenly. nCentroids is typically 256.
// Requires at least nCentroids training vectors.
func NewPQCodebook(vectors []Vector, nSubs, nCentroids int) *PQCodebook {
	if len(vectors) < nCentroids {
		return nil
	}
	dim := len(vectors[0].Embeddings)
	if dim%nSubs != 0 {
		return nil
	}
	subDim := dim / nSubs
	centroids := make([]float32, nSubs*nCentroids*subDim)
	for m := 0; m < nSubs; m++ {
		pqTrainSubspace(vectors, m, subDim, nCentroids, centroids[m*nCentroids*subDim:])
	}
	return &PQCodebook{NSubs: nSubs, NCentroids: nCentroids, SubDim: subDim, Centroids: centroids}
}

func pqTrainSubspace(vectors []Vector, sub, subDim, nCentroids int, centroids []float32) {
	offset := sub * subDim
	n := len(vectors)

	// Initialise centroids from random data points (no duplicates).
	perm := rand.Perm(n)
	for c := 0; c < nCentroids; c++ {
		copy(centroids[c*subDim:], vectors[perm[c]].Embeddings[offset:offset+subDim])
	}

	assignments := make([]int, n)
	newCentroids := make([]float32, len(centroids))
	counts := make([]int, nCentroids)

	for iter := 0; iter < 25; iter++ {
		changed := false
		for i, v := range vectors {
			sub_v := v.Embeddings[offset : offset+subDim]
			best, bestD := 0, float32(1e38)
			for c := 0; c < nCentroids; c++ {
				d := pqL2Sq(sub_v, centroids[c*subDim:(c+1)*subDim])
				if d < bestD {
					bestD, best = d, c
				}
			}
			if assignments[i] != best {
				assignments[i] = best
				changed = true
			}
		}
		if !changed {
			break
		}
		for i := range newCentroids {
			newCentroids[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}
		for i, v := range vectors {
			c := assignments[i]
			counts[c]++
			sub_v := v.Embeddings[offset : offset+subDim]
			for j, x := range sub_v {
				newCentroids[c*subDim+j] += x
			}
		}
		for c := 0; c < nCentroids; c++ {
			if counts[c] > 0 {
				inv := 1.0 / float32(counts[c])
				for j := 0; j < subDim; j++ {
					centroids[c*subDim+j] = newCentroids[c*subDim+j] * inv
				}
			}
		}
	}
}

func pqL2Sq(a, b []float32) float32 {
	var s float32
	for i := range a {
		d := a[i] - b[i]
		s += d * d
	}
	return s
}

// Encode quantizes a float32 vector into NSubs centroid indices (one byte each).
func (cb *PQCodebook) Encode(v []float32) []uint8 {
	codes := make([]uint8, cb.NSubs)
	for m := 0; m < cb.NSubs; m++ {
		sub := v[m*cb.SubDim : (m+1)*cb.SubDim]
		base := m * cb.NCentroids * cb.SubDim
		best, bestD := 0, float32(1e38)
		for c := 0; c < cb.NCentroids; c++ {
			d := pqL2Sq(sub, cb.Centroids[base+c*cb.SubDim:base+(c+1)*cb.SubDim])
			if d < bestD {
				bestD, best = d, c
			}
		}
		codes[m] = uint8(best)
	}
	return codes
}

// DistTable precomputes the distance from query to every centroid in every
// subspace. The returned slice has length NSubs*NCentroids.
func (cb *PQCodebook) DistTable(query []float32) []float32 {
	table := make([]float32, cb.NSubs*cb.NCentroids)
	for m := 0; m < cb.NSubs; m++ {
		sub := query[m*cb.SubDim : (m+1)*cb.SubDim]
		base := m * cb.NCentroids * cb.SubDim
		tbase := m * cb.NCentroids
		for c := 0; c < cb.NCentroids; c++ {
			table[tbase+c] = pqL2Sq(sub, cb.Centroids[base+c*cb.SubDim:base+(c+1)*cb.SubDim])
		}
	}
	return table
}

// DistPQ returns the approximate distance to a PQ-encoded vector using a
// precomputed distance table (from DistTable). This is O(NSubs) instead of O(Dim).
func (cb *PQCodebook) DistPQ(codes []uint8, table []float32) float32 {
	var s float32
	for m, c := range codes {
		s += table[m*cb.NCentroids+int(c)]
	}
	return s
}

// PQPool is a flat []uint8 backing array of PQ codes indexed by the same slot
// numbers as a VectorPool. No slot management — liveness is determined by
// the paired VectorPool.
type PQPool struct {
	data  []uint8
	nSubs int
}

func NewPQPool(nSubs, slots int) *PQPool {
	return &PQPool{data: make([]uint8, slots*nSubs), nSubs: nSubs}
}

func (p *PQPool) Set(idx int32, codes []uint8) {
	start := int(idx) * p.nSubs
	copy(p.data[start:], codes[:p.nSubs])
}

func (p *PQPool) Get(idx int32) []uint8 {
	start := int(idx) * p.nSubs
	return p.data[start : start+p.nSubs]
}

func (p *PQPool) Grow(n int) {
	need := n * p.nSubs
	for len(p.data) < need {
		p.data = append(p.data, 0)
	}
}
