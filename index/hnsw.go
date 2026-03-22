package index

import (
	"container/heap"
	"encoding/gob"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"

	"github.com/fayez/goatdb/core"
)

type candidate struct {
	id   string
	dist float32
}

type minHeap []candidate

func (h minHeap) Len() int            { return len(h) }
func (h minHeap) Less(i, j int) bool  { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)         { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() any           { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

type maxHeap []candidate

func (h maxHeap) Len() int            { return len(h) }
func (h maxHeap) Less(i, j int) bool  { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)         { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() any           { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

type hnswNode struct {
	id          string
	poolIdx     int32
	connections [][]string
}

type HNSWIndex struct {
	mu             sync.RWMutex
	dim            int
	M              int
	efConstruction int
	ef             int
	mL             float64
	nodes          map[string]*hnswNode
	entryPoint     string
	maxLayer       int
	distanceMetric core.DistanceMetric
	pool           *core.VectorPool
	codebook       *core.SQCodebook
	sqPool         *core.Int8VectorPool
	pqCodebook     *core.PQCodebook
	pqPool         *core.PQPool
}

func NewHNSWIndex(dim, M, efConstruction, ef int, metric core.DistanceMetric) *HNSWIndex {
	return &HNSWIndex{
		dim:            dim,
		M:              M,
		efConstruction: efConstruction,
		ef:             ef,
		mL:             1.0 / math.Log(float64(M)),
		nodes:          make(map[string]*hnswNode),
		maxLayer:       -1,
		distanceMetric: metric,
		pool:           core.NewVectorPool(dim, 64),
	}
}

// Train computes the SQ codebook from training vectors and quantizes all
// existing pool vectors into the int8 pool. Must be called before AddVector
// for maximum benefit (subsequent AddVector calls also quantize into sqPool).
func (h *HNSWIndex) Train(vectors []core.Vector) {
	h.mu.Lock()
	defer h.mu.Unlock()

	// Scalar quantization (SQ): global min/max float32→int8.
	cb := core.NewSQCodebook(vectors)
	if cb == nil {
		return
	}
	h.codebook = cb
	h.sqPool = core.NewInt8VectorPool(h.dim, max(h.pool.Slots(), 64))
	for _, node := range h.nodes {
		h.sqPool.Set(node.poolIdx, cb.Quantize(h.pool.Get(node.poolIdx)))
	}

	// Product quantization (PQ): split into subspaces, 256 centroids each.
	// nSubs chosen so subDim=4 (fast per-subspace k-means); fall back to
	// smaller divisor if dim is not divisible.
	nSubs := h.dim / 4
	if nSubs < 1 {
		nSubs = 1
	}
	for nSubs > 1 && h.dim%nSubs != 0 {
		nSubs--
	}
	pq := core.NewPQCodebook(vectors, nSubs, 256)
	if pq == nil {
		return
	}
	h.pqCodebook = pq
	h.pqPool = core.NewPQPool(nSubs, max(h.pool.Slots(), 64))
	for _, node := range h.nodes {
		h.pqPool.Set(node.poolIdx, pq.Encode(h.pool.Get(node.poolIdx)))
	}
}

func (h *HNSWIndex) randomLevel() int {
	return int(-math.Log(rand.Float64()) * h.mL)
}

func (h *HNSWIndex) dist(a, b []float32) float32 {
	return core.DistSlices(a, b, h.distanceMetric)
}

// searchLayer runs the beam search at a single HNSW layer.
// Priority for distance computation: PQ table (fastest) → SQ int8 → float32 exact.
// The caller is responsible for float32 re-ranking after this returns.
func (h *HNSWIndex) searchLayer(query []float32, queryInt8 []int8, distTable []float32, eps []candidate, ef, layer int) []candidate {
	visited := make(map[string]bool, ef*2)

	cands := &minHeap{}
	W := &maxHeap{}

	for _, ep := range eps {
		heap.Push(cands, ep)
		heap.Push(W, ep)
		visited[ep.id] = true
	}

	for cands.Len() > 0 {
		c := heap.Pop(cands).(candidate)
		f := (*W)[0]
		if c.dist > f.dist {
			break
		}
		node := h.nodes[c.id]
		if layer >= len(node.connections) {
			continue
		}
		for _, nbID := range node.connections[layer] {
			if visited[nbID] {
				continue
			}
			visited[nbID] = true
			nb := h.nodes[nbID]
			var d float32
			switch {
			case distTable != nil:
				d = h.pqCodebook.DistPQ(h.pqPool.Get(nb.poolIdx), distTable)
			case queryInt8 != nil:
				d = h.codebook.DistInt8(queryInt8, h.sqPool.Get(nb.poolIdx))
			default:
				d = h.dist(query, h.pool.Get(nb.poolIdx))
			}
			f = (*W)[0]
			if d < f.dist || W.Len() < ef {
				heap.Push(cands, candidate{nbID, d})
				heap.Push(W, candidate{nbID, d})
				if W.Len() > ef {
					heap.Pop(W)
				}
			}
		}
	}

	result := make([]candidate, W.Len())
	for i := len(result) - 1; i >= 0; i-- {
		result[i] = heap.Pop(W).(candidate)
	}
	return result
}

func (h *HNSWIndex) selectNeighborsHeuristic(query []float32, candidates []candidate, M int) []candidate {
	if len(candidates) <= M {
		return candidates
	}
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].dist < candidates[j].dist
	})
	result := make([]candidate, 0, M)
	for _, c := range candidates {
		if len(result) >= M {
			break
		}
		dominated := false
		cNode := h.nodes[c.id]
		for _, r := range result {
			rNode := h.nodes[r.id]
			if h.dist(h.pool.Get(rNode.poolIdx), h.pool.Get(cNode.poolIdx)) < c.dist {
				dominated = true
				break
			}
		}
		if !dominated {
			result = append(result, c)
		}
	}
	return result
}

func (h *HNSWIndex) AddVector(id string, vector core.Vector) {
	h.mu.Lock()
	defer h.mu.Unlock()
	level := h.randomLevel()
	poolIdx := h.pool.Add(vector.Embeddings)

	if h.codebook != nil {
		h.sqPool.Grow(int(poolIdx) + 1)
		h.sqPool.Set(poolIdx, h.codebook.Quantize(vector.Embeddings))
	}
	if h.pqCodebook != nil {
		h.pqPool.Grow(int(poolIdx) + 1)
		h.pqPool.Set(poolIdx, h.pqCodebook.Encode(vector.Embeddings))
	}

	node := &hnswNode{
		id:          id,
		poolIdx:     poolIdx,
		connections: make([][]string, level+1),
	}
	for i := range node.connections {
		node.connections[i] = []string{}
	}
	h.nodes[id] = node

	if h.maxLayer == -1 {
		h.entryPoint = id
		h.maxLayer = level
		return
	}

	q := vector.Embeddings
	var qInt8 []int8
	if h.codebook != nil {
		qInt8 = h.codebook.Quantize(q)
	}
	var distTable []float32
	if h.pqCodebook != nil {
		distTable = h.pqCodebook.DistTable(q)
	}

	ep := []candidate{{h.entryPoint, h.dist(q, h.pool.Get(h.nodes[h.entryPoint].poolIdx))}}

	for layer := h.maxLayer; layer > level; layer-- {
		result := h.searchLayer(q, qInt8, distTable, ep, 1, layer)
		ep = result[:1]
	}

	for layer := min(level, h.maxLayer); layer >= 0; layer-- {
		mMax := h.M
		if layer == 0 {
			mMax = h.M * 2
		}
		candidates := h.searchLayer(q, qInt8, distTable, ep, h.efConstruction, layer)

		// re-rank candidates with exact float32 distances before neighbor selection
		if distTable != nil || qInt8 != nil {
			for i := range candidates {
				candidates[i].dist = h.dist(q, h.pool.Get(h.nodes[candidates[i].id].poolIdx))
			}
			sort.Slice(candidates, func(i, j int) bool { return candidates[i].dist < candidates[j].dist })
		}

		neighbors := h.selectNeighborsHeuristic(q, candidates, mMax)

		node.connections[layer] = make([]string, len(neighbors))
		for i, nb := range neighbors {
			node.connections[layer][i] = nb.id
		}

		for _, nb := range neighbors {
			nbNode := h.nodes[nb.id]
			if layer >= len(nbNode.connections) {
				continue
			}
			nbNode.connections[layer] = append(nbNode.connections[layer], id)
			if len(nbNode.connections[layer]) > mMax {
				conns := nbNode.connections[layer]
				pruned := make([]candidate, len(conns))
				for i, connID := range conns {
					conn := h.nodes[connID]
					pruned[i] = candidate{connID, h.dist(h.pool.Get(nbNode.poolIdx), h.pool.Get(conn.poolIdx))}
				}
				for i := 1; i < len(pruned); i++ {
					for j := i; j > 0 && pruned[j].dist < pruned[j-1].dist; j-- {
						pruned[j], pruned[j-1] = pruned[j-1], pruned[j]
					}
				}
				pruned = pruned[:mMax]
				nbNode.connections[layer] = make([]string, mMax)
				for i, p := range pruned {
					nbNode.connections[layer][i] = p.id
				}
			}
		}
		ep = candidates
	}

	if level > h.maxLayer {
		h.maxLayer = level
		h.entryPoint = id
	}
}

func (h *HNSWIndex) GetVector(id string) (core.Vector, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	node, ok := h.nodes[id]
	if !ok {
		return core.Vector{}, false
	}
	emb := make([]float32, h.dim)
	copy(emb, h.pool.Get(node.poolIdx))
	return core.Vector{Embeddings: emb}, true
}

func (h *HNSWIndex) DeleteVector(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	node, exists := h.nodes[id]
	if !exists {
		return false
	}
	for layer, conns := range node.connections {
		for _, nbID := range conns {
			nb := h.nodes[nbID]
			if layer >= len(nb.connections) {
				continue
			}
			updated := nb.connections[layer][:0]
			for _, c := range nb.connections[layer] {
				if c != id {
					updated = append(updated, c)
				}
			}
			nb.connections[layer] = updated
		}
	}
	h.pool.Free(node.poolIdx)
	delete(h.nodes, id)

	if h.entryPoint == id {
		h.maxLayer = -1
		for newID, n := range h.nodes {
			lvl := len(n.connections) - 1
			if lvl > h.maxLayer {
				h.maxLayer = lvl
				h.entryPoint = newID
			}
		}
	}
	return true
}

func (h *HNSWIndex) Search(query core.Vector, topK int) []core.SearchResult {
	h.mu.RLock()
	defer h.mu.RUnlock()
	if h.maxLayer == -1 {
		return nil
	}

	q := query.Embeddings
	var qInt8 []int8
	if h.codebook != nil {
		qInt8 = h.codebook.Quantize(q)
	}
	var distTable []float32
	if h.pqCodebook != nil {
		distTable = h.pqCodebook.DistTable(q)
	}

	ep := []candidate{{h.entryPoint, h.dist(q, h.pool.Get(h.nodes[h.entryPoint].poolIdx))}}

	for layer := h.maxLayer; layer > 0; layer-- {
		result := h.searchLayer(q, qInt8, distTable, ep, 1, layer)
		ep = result[:1]
	}

	candidates := h.searchLayer(q, qInt8, distTable, ep, max(h.ef, topK), 0)

	// final re-rank with exact float32 distances
	if distTable != nil || qInt8 != nil {
		for i := range candidates {
			candidates[i].dist = h.dist(q, h.pool.Get(h.nodes[candidates[i].id].poolIdx))
		}
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].dist < candidates[j].dist })
	}

	results := make([]core.SearchResult, 0, topK)
	for i := 0; i < topK && i < len(candidates); i++ {
		results = append(results, core.SearchResult{
			Id:       candidates[i].id,
			Distance: candidates[i].dist,
		})
	}
	return results
}

type hnswNodeState struct {
	Embeddings  []float32
	Connections [][]string
}

type hnswState struct {
	Dim            int
	M              int
	EfConstruction int
	Ef             int
	ML             float64
	DistanceMetric core.DistanceMetric
	EntryPoint     string
	MaxLayer       int
	Nodes          map[string]hnswNodeState
	SQMin          float32
	SQScale        float32
	HasSQ          bool
	PQNSubs        int
	PQNCentroids   int
	PQCentroids    []float32
	HasPQ          bool
}

func (h *HNSWIndex) Save(path string) error {
	h.mu.RLock()
	defer h.mu.RUnlock()
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	nodes := make(map[string]hnswNodeState, len(h.nodes))
	for id, node := range h.nodes {
		emb := make([]float32, h.dim)
		copy(emb, h.pool.Get(node.poolIdx))
		nodes[id] = hnswNodeState{
			Embeddings:  emb,
			Connections: node.connections,
		}
	}

	state := hnswState{
		Dim:            h.dim,
		M:              h.M,
		EfConstruction: h.efConstruction,
		Ef:             h.ef,
		ML:             h.mL,
		DistanceMetric: h.distanceMetric,
		EntryPoint:     h.entryPoint,
		MaxLayer:       h.maxLayer,
		Nodes:          nodes,
	}
	if h.codebook != nil {
		state.HasSQ = true
		state.SQMin = h.codebook.Min
		state.SQScale = h.codebook.Scale
	}
	if h.pqCodebook != nil {
		state.HasPQ = true
		state.PQNSubs = h.pqCodebook.NSubs
		state.PQNCentroids = h.pqCodebook.NCentroids
		state.PQCentroids = h.pqCodebook.Centroids
	}
	return gob.NewEncoder(file).Encode(state)
}

func (h *HNSWIndex) Load(path string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	var s hnswState
	if err := gob.NewDecoder(file).Decode(&s); err != nil {
		return err
	}
	h.dim = s.Dim
	h.M = s.M
	h.efConstruction = s.EfConstruction
	h.ef = s.Ef
	h.mL = s.ML
	h.distanceMetric = s.DistanceMetric
	h.entryPoint = s.EntryPoint
	h.maxLayer = s.MaxLayer
	h.pool = core.NewVectorPool(s.Dim, len(s.Nodes))
	h.nodes = make(map[string]*hnswNode, len(s.Nodes))
	for id, ns := range s.Nodes {
		poolIdx := h.pool.Add(ns.Embeddings)
		h.nodes[id] = &hnswNode{
			id:          id,
			poolIdx:     poolIdx,
			connections: ns.Connections,
		}
	}

	if s.HasSQ {
		h.codebook = &core.SQCodebook{Min: s.SQMin, Scale: s.SQScale, Dim: s.Dim}
		h.sqPool = core.NewInt8VectorPool(s.Dim, h.pool.Slots())
		for _, node := range h.nodes {
			h.sqPool.Set(node.poolIdx, h.codebook.Quantize(h.pool.Get(node.poolIdx)))
		}
	}
	if s.HasPQ {
		subDim := s.Dim / s.PQNSubs
		h.pqCodebook = &core.PQCodebook{
			NSubs:      s.PQNSubs,
			NCentroids: s.PQNCentroids,
			SubDim:     subDim,
			Centroids:  s.PQCentroids,
		}
		h.pqPool = core.NewPQPool(s.PQNSubs, h.pool.Slots())
		for _, node := range h.nodes {
			h.pqPool.Set(node.poolIdx, h.pqCodebook.Encode(h.pool.Get(node.poolIdx)))
		}
	}
	return nil
}
