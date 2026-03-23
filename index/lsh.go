package index

import (
	"container/heap"
	"encoding/gob"
	"math/rand"
	"os"
	"sync"

	"github.com/fayez/goatdb/core"
)

type lshTable struct {
	hyperplanes [][]float32 // [K][dim]
	buckets     map[uint64][]int32
}

// LSHIndex is a multi-probe random-hyperplane LSH index.
// Vectors are stored in a contiguous VectorPool; hash computation uses the
// AVX2-accelerated dot product from core when available.
type LSHIndex struct {
	mu             sync.RWMutex
	dim            int
	L              int
	K              int
	tables         []lshTable
	pool           *core.VectorPool
	slotToID       map[int32]string
	idToSlot       map[string]int32
	distanceMetric core.DistanceMetric
	perturbMasks   []uint64
}

// multiProbeMasks returns a set of bit masks to XOR against the primary hash
// bucket key during multi-probe search. Includes all 1-bit and 2-bit flips.
func multiProbeMasks(K int) []uint64 {
	var masks []uint64
	for i := 0; i < K; i++ {
		masks = append(masks, 1<<uint(i))
	}
	for i := 0; i < K; i++ {
		for j := i + 1; j < K; j++ {
			masks = append(masks, (1<<uint(i))|(1<<uint(j)))
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
		tables[i] = lshTable{hyperplanes: hps, buckets: make(map[uint64][]int32)}
	}
	return &LSHIndex{
		dim:            dim,
		L:              L,
		K:              K,
		tables:         tables,
		pool:           core.NewVectorPool(dim, 64),
		slotToID:       make(map[int32]string),
		idToSlot:       make(map[string]int32),
		distanceMetric: metric,
		perturbMasks:   multiProbeMasks(K),
	}
}

// hashVec computes the K-bit bucket key for vector v in table t.
// Each bit is the sign of the dot product with the corresponding hyperplane.
// Uses core.DistSlices (AVX2 dispatch) for each dot product.
func (l *LSHIndex) hashVec(t *lshTable, v []float32) uint64 {
	var h uint64
	for i, hp := range t.hyperplanes {
		if core.DistSlices(v, hp, core.DotProduct) >= 0 {
			h |= 1 << uint(i)
		}
	}
	return h
}

func (l *LSHIndex) AddVector(id string, vector core.Vector) {
	l.mu.Lock()
	defer l.mu.Unlock()

	slot := l.pool.Add(vector.Embeddings)
	l.idToSlot[id] = slot
	l.slotToID[slot] = id
	for i := range l.tables {
		h := l.hashVec(&l.tables[i], vector.Embeddings)
		l.tables[i].buckets[h] = append(l.tables[i].buckets[h], slot)
	}
}

func (l *LSHIndex) GetVector(id string) (core.Vector, bool) {
	l.mu.RLock()
	defer l.mu.RUnlock()

	slot, ok := l.idToSlot[id]
	if !ok {
		return core.Vector{}, false
	}
	emb := make([]float32, l.dim)
	copy(emb, l.pool.Get(slot))
	return core.Vector{Embeddings: emb}, true
}

func (l *LSHIndex) DeleteVector(id string) bool {
	l.mu.Lock()
	defer l.mu.Unlock()

	slot, ok := l.idToSlot[id]
	if !ok {
		return false
	}
	emb := l.pool.Get(slot) // read before freeing
	for i := range l.tables {
		h := l.hashVec(&l.tables[i], emb)
		bucket := l.tables[i].buckets[h]
		for j, s := range bucket {
			if s == slot {
				last := len(bucket) - 1
				bucket[j] = bucket[last]
				l.tables[i].buckets[h] = bucket[:last]
				break
			}
		}
	}
	l.pool.Free(slot)
	delete(l.idToSlot, id)
	delete(l.slotToID, slot)
	return true
}

// maxCandidateFrac caps the LSH candidate set as a fraction of the index size
// to prevent degeneration into a full scan on large datasets.
// 40% coverage: enough for good recall, bounded to avoid linear-scan cost.
const maxCandidateFrac = 0.40

func (l *LSHIndex) Search(query core.Vector, topK int) []core.SearchResult {
	l.mu.RLock()
	defer l.mu.RUnlock()

	maxCands := int(float64(l.pool.Len()) * maxCandidateFrac)
	if maxCands < 1000 {
		maxCands = 1000
	}
	seen := make(map[int32]bool)
	candidates := make([]int32, 0, maxCands)
	q := query.Embeddings

outer:
	for i := range l.tables {
		h := l.hashVec(&l.tables[i], q)
		for _, slot := range l.tables[i].buckets[h] {
			if !seen[slot] {
				seen[slot] = true
				candidates = append(candidates, slot)
				if len(candidates) >= maxCands {
					break outer
				}
			}
		}
		for _, mask := range l.perturbMasks {
			for _, slot := range l.tables[i].buckets[h^mask] {
				if !seen[slot] {
					seen[slot] = true
					candidates = append(candidates, slot)
					if len(candidates) >= maxCands {
						break outer
					}
				}
			}
		}
	}

	rh := &resultHeap{}
	heap.Init(rh)
	for _, slot := range candidates {
		emb := l.pool.Get(slot)
		dist := core.DistSlices(q, emb, l.distanceMetric)
		result := core.SearchResult{Id: l.slotToID[slot], Distance: dist}
		if rh.Len() < topK {
			heap.Push(rh, result)
		} else if dist < (*rh)[0].Distance {
			heap.Pop(rh)
			heap.Push(rh, result)
		}
	}
	return []core.SearchResult(*rh)
}

// ---- persistence ----

type lshState struct {
	Dim            int
	L              int
	K              int
	DistanceMetric core.DistanceMetric
	Hyperplanes    [][][]float32
	Buckets        []map[uint64][]int32
	Vectors        map[string][]float32
}

func (l *LSHIndex) Save(path string) error {
	l.mu.RLock()
	defer l.mu.RUnlock()

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	hps := make([][][]float32, l.L)
	buckets := make([]map[uint64][]int32, l.L)
	for i, t := range l.tables {
		hps[i] = t.hyperplanes
		buckets[i] = t.buckets
	}
	vecs := make(map[string][]float32, len(l.idToSlot))
	for id, slot := range l.idToSlot {
		emb := make([]float32, l.dim)
		copy(emb, l.pool.Get(slot))
		vecs[id] = emb
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
	l.mu.Lock()
	defer l.mu.Unlock()

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
	l.perturbMasks = multiProbeMasks(s.K)

	l.pool = core.NewVectorPool(s.Dim, max(len(s.Vectors), 64))
	l.idToSlot = make(map[string]int32, len(s.Vectors))
	l.slotToID = make(map[int32]string, len(s.Vectors))
	for id, emb := range s.Vectors {
		slot := l.pool.Add(emb)
		l.idToSlot[id] = slot
		l.slotToID[slot] = id
	}

	l.tables = make([]lshTable, s.L)
	for i := range l.tables {
		l.tables[i] = lshTable{hyperplanes: s.Hyperplanes[i], buckets: s.Buckets[i]}
	}
	return nil
}
