package index

import (
	"container/heap"
	"encoding/gob"
	"math/rand"
	"os"

	"github.com/fayez/goatdb/core"
)

type lshTable struct {
	hyperplanes [][]float32
	buckets     map[uint64][]string
}

type LSHIndex struct {
	dim            int
	L              int
	K              int
	tables         []lshTable
	vectors        map[string]core.Vector
	distanceMetric core.DistanceMetric
	perturbMasks   []uint64
}

func multiProbeMasks(K int) []uint64 {
	var masks []uint64
	for i := 0; i < K; i++ {
		masks = append(masks, 1<<uint(i))
	}
	if K <= 6 {
		for i := 0; i < K; i++ {
			for j := i + 1; j < K; j++ {
				masks = append(masks, (1<<uint(i))|(1<<uint(j)))
			}
		}
	}
	return masks
}

func NewLSHIndex(dim, L, K int, metric core.DistanceMetric) *LSHIndex {
	tables := make([]lshTable, L)
	for i := range tables {
		hps := make([][]float32, K)
		for j := range hps {
			hp := make([]float32, dim)
			for k := range hp {
				hp[k] = float32(rand.NormFloat64())
			}
			hps[j] = hp
		}
		tables[i] = lshTable{hyperplanes: hps, buckets: make(map[uint64][]string)}
	}
	return &LSHIndex{
		dim:            dim,
		L:              L,
		K:              K,
		tables:         tables,
		vectors:        make(map[string]core.Vector),
		distanceMetric: metric,
		perturbMasks:   multiProbeMasks(K),
	}
}

func (l *LSHIndex) hashVec(t lshTable, v []float32) uint64 {
	var h uint64
	for i, hp := range t.hyperplanes {
		var dot float32
		for j := range v {
			dot += v[j] * hp[j]
		}
		if dot >= 0 {
			h |= 1 << uint(i)
		}
	}
	return h
}

func (l *LSHIndex) AddVector(id string, vector core.Vector) {
	l.vectors[id] = vector
	for i := range l.tables {
		h := l.hashVec(l.tables[i], vector.Embeddings)
		l.tables[i].buckets[h] = append(l.tables[i].buckets[h], id)
	}
}

func (l *LSHIndex) GetVector(id string) (core.Vector, bool) {
	v, ok := l.vectors[id]
	return v, ok
}

func (l *LSHIndex) DeleteVector(id string) bool {
	if _, exists := l.vectors[id]; !exists {
		return false
	}
	v := l.vectors[id]
	delete(l.vectors, id)
	for i := range l.tables {
		h := l.hashVec(l.tables[i], v.Embeddings)
		bucket := l.tables[i].buckets[h]
		for j, vid := range bucket {
			if vid == id {
				l.tables[i].buckets[h] = append(bucket[:j], bucket[j+1:]...)
				break
			}
		}
	}
	return true
}

func (l *LSHIndex) Search(query core.Vector, topK int) []core.SearchResult {
	seen := make(map[string]bool)
	var candidates []string
	for i := range l.tables {
		h := l.hashVec(l.tables[i], query.Embeddings)
		for _, id := range l.tables[i].buckets[h] {
			if !seen[id] {
				seen[id] = true
				candidates = append(candidates, id)
			}
		}
		for _, mask := range l.perturbMasks {
			for _, id := range l.tables[i].buckets[h^mask] {
				if !seen[id] {
					seen[id] = true
					candidates = append(candidates, id)
				}
			}
		}
	}

	rh := &resultHeap{}
	heap.Init(rh)
	for _, id := range candidates {
		v := l.vectors[id]
		dist := query.Distance(&v, l.distanceMetric)
		result := core.SearchResult{Id: id, Distance: dist}
		if rh.Len() < topK {
			heap.Push(rh, result)
		} else if dist < (*rh)[0].Distance {
			heap.Pop(rh)
			heap.Push(rh, result)
		}
	}
	return []core.SearchResult(*rh)
}

type lshState struct {
	Dim            int
	L              int
	K              int
	DistanceMetric core.DistanceMetric
	Hyperplanes    [][][]float32
	Buckets        []map[uint64][]string
	Vectors        map[string][]float32
}

func (l *LSHIndex) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	hps := make([][][]float32, l.L)
	buckets := make([]map[uint64][]string, l.L)
	for i, t := range l.tables {
		hps[i] = t.hyperplanes
		buckets[i] = t.buckets
	}
	vecs := make(map[string][]float32, len(l.vectors))
	for id, v := range l.vectors {
		vecs[id] = v.Embeddings
	}
	return gob.NewEncoder(file).Encode(lshState{
		Dim:            l.dim,
		L:              l.L,
		K:              l.K,
		DistanceMetric: l.distanceMetric,
		Hyperplanes:    hps,
		Buckets:        buckets,
		Vectors:        vecs,
	})
}

func (l *LSHIndex) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	var s lshState
	if err := gob.NewDecoder(file).Decode(&s); err != nil {
		return err
	}
	l.dim = s.Dim
	l.L = s.L
	l.K = s.K
	l.distanceMetric = s.DistanceMetric
	l.tables = make([]lshTable, s.L)
	for i := range l.tables {
		l.tables[i] = lshTable{hyperplanes: s.Hyperplanes[i], buckets: s.Buckets[i]}
	}
	l.vectors = make(map[string]core.Vector, len(s.Vectors))
	for id, emb := range s.Vectors {
		l.vectors[id] = core.Vector{Embeddings: emb}
	}
	l.perturbMasks = multiProbeMasks(l.K)
	return nil
}
