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
	slot int32
	dist float32
}

type minHeap []candidate

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].dist < h[j].dist }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *minHeap) Pop() any          { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

type maxHeap []candidate

func (h maxHeap) Len() int           { return len(h) }
func (h maxHeap) Less(i, j int) bool { return h[i].dist > h[j].dist }
func (h maxHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *maxHeap) Push(x any)        { *h = append(*h, x.(candidate)) }
func (h *maxHeap) Pop() any          { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

type hnswNode struct {
	id          string
	poolIdx     int32
	connections [][]int32
}

type HNSWIndex struct {
	mu             sync.RWMutex
	dim            int
	M              int
	efConstruction int
	ef             int
	mL             float64
	nodes          map[string]*hnswNode
	slotToNode     []*hnswNode
	entrySlot      int32
	maxLayer       int
	distanceMetric core.DistanceMetric
	metric         core.Metric
	pool           *core.VectorPool
	codebook       *core.SQCodebook
	sqPool         *core.Int8VectorPool
	pqCodebook     *core.PQCodebook
	pqPool         *core.PQPool
	// visitedPool reuses []byte visited arrays across searchLayer calls.
	// Flat array lookup is O(1) at ~2ns vs map at ~50ns — at efConstruction=200
	// and M=24 this saves ~690µs per AddVector, 11+ minutes at 1M.
	visitedPool sync.Pool
}

func NewHNSWIndex(dim, M, efConstruction, ef int, metric core.DistanceMetric) *HNSWIndex {
	h := &HNSWIndex{
		dim:            dim,
		M:              M,
		efConstruction: efConstruction,
		ef:             ef,
		mL:             1.0 / math.Log(float64(M)),
		nodes:          make(map[string]*hnswNode),
		maxLayer:       -1,
		distanceMetric: metric,
		metric:         core.ResolveMetric(metric),
		pool:           core.NewVectorPool(dim, 64),
	}
	h.visitedPool.New = func() any { b := make([]byte, 0); return &b }
	return h
}

func (h *HNSWIndex) Train(vectors []core.Vector) {
	h.mu.Lock()
	defer h.mu.Unlock()

	vectors = prepareVectors(h.metric, vectors, hnswMaxTrainVecs)
	cb := core.NewSQCodebook(vectors)
	if cb == nil {
		return
	}
	h.codebook = cb
	h.sqPool = core.NewInt8VectorPool(h.dim, max(h.pool.Slots(), 64))
	for _, node := range h.nodes {
		h.sqPool.Set(node.poolIdx, cb.Quantize(h.pool.Get(node.poolIdx)))
	}

	nSubs := h.dim / 8
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

// hnswMaxTrainVecs caps the normalized copies made for codebook training.
const hnswMaxTrainVecs = 100_000

func (h *HNSWIndex) randomLevel() int {
	return int(-math.Log(rand.Float64()) * h.mL)
}

func (h *HNSWIndex) dist(a, b []float32) float32 {
	return h.metric.Dist(a, b)
}

func (h *HNSWIndex) growSlotToNode(slot int32) {
	for int(slot) >= len(h.slotToNode) {
		h.slotToNode = append(h.slotToNode, nil)
	}
}

func (h *HNSWIndex) slotDist(q []float32, qInt8 []int8, slot int32) float32 {
	if qInt8 != nil {
		return h.codebook.DistInt8(qInt8, h.sqPool.Get(slot))
	}
	return h.dist(q, h.pool.Get(slot))
}

func (h *HNSWIndex) searchLayer(query []float32, queryInt8 []int8, eps []candidate, ef, layer int) []candidate {
	nSlots := h.pool.Slots()
	vbp := h.visitedPool.Get().(*[]byte)
	vb := *vbp
	if cap(vb) < nSlots {
		vb = make([]byte, nSlots)
	} else {
		vb = vb[:nSlots]
	}
	dirty := make([]int32, 0, ef*4)

	cands := &minHeap{}
	W := &maxHeap{}

	for _, ep := range eps {
		heap.Push(cands, ep)
		heap.Push(W, ep)
		if int(ep.slot) < len(vb) && vb[ep.slot] == 0 {
			vb[ep.slot] = 1
			dirty = append(dirty, ep.slot)
		}
	}

	for cands.Len() > 0 {
		c := heap.Pop(cands).(candidate)
		if c.dist > (*W)[0].dist {
			break
		}
		node := h.slotToNode[c.slot]
		if layer >= len(node.connections) {
			continue
		}
		for _, nbSlot := range node.connections[layer] {
			if int(nbSlot) < len(vb) && vb[nbSlot] != 0 {
				continue
			}
			if int(nbSlot) < len(vb) {
				vb[nbSlot] = 1
				dirty = append(dirty, nbSlot)
			}
			d := h.slotDist(query, queryInt8, nbSlot)
			if d < (*W)[0].dist || W.Len() < ef {
				heap.Push(cands, candidate{nbSlot, d})
				heap.Push(W, candidate{nbSlot, d})
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

	for _, s := range dirty {
		if int(s) < len(vb) {
			vb[s] = 0
		}
	}
	*vbp = vb
	h.visitedPool.Put(vbp)

	return result
}

func (h *HNSWIndex) selectNeighbors(candidates []candidate, M int) []candidate {
	if len(candidates) <= M {
		return candidates
	}
	return candidates[:M]
}
func (h *HNSWIndex) AddVector(id string, vector core.Vector) {
	level := h.randomLevel()

	h.mu.Lock()
	defer h.mu.Unlock()

	poolIdx := h.pool.Add(vector.Embeddings)
	q := h.pool.Get(poolIdx)
	h.metric.PrepareInPlace(q)
	if h.codebook != nil {
		h.sqPool.Grow(int(poolIdx) + 1)
		h.sqPool.Set(poolIdx, h.codebook.Quantize(q))
	}
	if h.pqCodebook != nil {
		h.pqPool.Grow(int(poolIdx) + 1)
		h.pqPool.Set(poolIdx, h.pqCodebook.Encode(q))
	}
	node := &hnswNode{
		id:          id,
		poolIdx:     poolIdx,
		connections: make([][]int32, level+1),
	}
	for i := range node.connections {
		node.connections[i] = []int32{}
	}
	h.nodes[id] = node
	h.growSlotToNode(poolIdx)
	h.slotToNode[poolIdx] = node

	if h.maxLayer == -1 {
		h.entrySlot = poolIdx
		h.maxLayer = level
		return
	}

	var qInt8 []int8
	if h.codebook != nil {
		qInt8 = h.codebook.Quantize(q)
	}

	ep := []candidate{{h.entrySlot, h.slotDist(q, qInt8, h.entrySlot)}}

	for layer := h.maxLayer; layer > level; layer-- {
		result := h.searchLayer(q, qInt8, ep, 1, layer)
		ep = result[:1]
	}

	for layer := min(level, h.maxLayer); layer >= 0; layer-- {
		mMax := h.M
		if layer == 0 {
			mMax = h.M * 2
		}
		candidates := h.searchLayer(q, qInt8, ep, h.efConstruction, layer)
		if qInt8 != nil {
			for i := range candidates {
				candidates[i].dist = h.dist(q, h.pool.Get(candidates[i].slot))
			}
			sort.Slice(candidates, func(i, j int) bool { return candidates[i].dist < candidates[j].dist })
		}
		neighbors := h.selectNeighbors(candidates, mMax)
		node.connections[layer] = make([]int32, len(neighbors))
		for i, nb := range neighbors {
			node.connections[layer][i] = nb.slot
		}
		for _, nb := range neighbors {
			nbNode := h.slotToNode[nb.slot]
			if layer >= len(nbNode.connections) {
				continue
			}
			conns := nbNode.connections[layer]
			if len(conns) < mMax {
				nbNode.connections[layer] = append(conns, poolIdx)
				continue
			}

			nbVec := h.pool.Get(nb.slot)
			newDist := h.dist(nbVec, h.pool.Get(poolIdx))
			worstIdx, worstDist := -1, float32(-math.MaxFloat32)
			for ci, connSlot := range conns {
				d := h.dist(nbVec, h.pool.Get(connSlot))
				if d > worstDist {
					worstDist, worstIdx = d, ci
				}
			}
			if worstIdx >= 0 && newDist < worstDist {
				nbNode.connections[layer][worstIdx] = poolIdx
			}
		}
		ep = candidates
	}

	if level > h.maxLayer {
		h.maxLayer = level
		h.entrySlot = poolIdx
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
	slot := node.poolIdx
	for layer, conns := range node.connections {
		for _, nbSlot := range conns {
			nb := h.slotToNode[nbSlot]
			if layer >= len(nb.connections) {
				continue
			}
			updated := nb.connections[layer][:0]
			for _, s := range nb.connections[layer] {
				if s != slot {
					updated = append(updated, s)
				}
			}
			nb.connections[layer] = updated
		}
	}
	h.pool.Free(slot)
	h.slotToNode[slot] = nil
	delete(h.nodes, id)

	if h.entrySlot == slot {
		h.maxLayer = -1
		for _, n := range h.nodes {
			lvl := len(n.connections) - 1
			if lvl > h.maxLayer {
				h.maxLayer = lvl
				h.entrySlot = n.poolIdx
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

	q := h.metric.Prepare(query.Embeddings)
	var qInt8 []int8
	if h.codebook != nil {
		qInt8 = h.codebook.Quantize(q)
	}

	ep := []candidate{{h.entrySlot, h.slotDist(q, qInt8, h.entrySlot)}}

	for layer := h.maxLayer; layer > 0; layer-- {
		result := h.searchLayer(q, qInt8, ep, 1, layer)
		ep = result[:1]
	}

	candidates := h.searchLayer(q, qInt8, ep, max(h.ef, topK), 0)

	if qInt8 != nil {
		for i := range candidates {
			candidates[i].dist = h.dist(q, h.pool.Get(candidates[i].slot))
		}
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].dist < candidates[j].dist })
	}

	results := make([]core.SearchResult, 0, topK)
	for i := 0; i < topK && i < len(candidates); i++ {
		results = append(results, core.SearchResult{
			Id:       h.slotToNode[candidates[i].slot].id,
			Distance: h.metric.Finalize(candidates[i].dist),
		})
	}
	return results
}

// ---- persistence ----

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
	Normalized     bool
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
		conns := make([][]string, len(node.connections))
		for l, layer := range node.connections {
			conns[l] = make([]string, len(layer))
			for i, s := range layer {
				if n := h.slotToNode[s]; n != nil {
					conns[l][i] = n.id
				}
			}
		}
		nodes[id] = hnswNodeState{Embeddings: emb, Connections: conns}
	}

	entryID := ""
	if h.maxLayer >= 0 && int(h.entrySlot) < len(h.slotToNode) && h.slotToNode[h.entrySlot] != nil {
		entryID = h.slotToNode[h.entrySlot].id
	}

	state := hnswState{
		Dim:            h.dim,
		M:              h.M,
		EfConstruction: h.efConstruction,
		Ef:             h.ef,
		ML:             h.mL,
		DistanceMetric: h.distanceMetric,
		EntryPoint:     entryID,
		MaxLayer:       h.maxLayer,
		Nodes:          nodes,
		Normalized:     h.metric.Normalizes(),
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
	h.metric = core.ResolveMetric(s.DistanceMetric)
	h.maxLayer = s.MaxLayer
	h.pool = core.NewVectorPool(s.Dim, len(s.Nodes))
	h.nodes = make(map[string]*hnswNode, len(s.Nodes))
	h.slotToNode = make([]*hnswNode, 0, len(s.Nodes))

	for id, ns := range s.Nodes {
		poolIdx := h.pool.Add(ns.Embeddings)
		if !s.Normalized {
			h.metric.PrepareInPlace(h.pool.Get(poolIdx))
		}
		node := &hnswNode{id: id, poolIdx: poolIdx}
		h.nodes[id] = node
		h.growSlotToNode(poolIdx)
		h.slotToNode[poolIdx] = node
	}

	for id, ns := range s.Nodes {
		node := h.nodes[id]
		node.connections = make([][]int32, len(ns.Connections))
		for l, layer := range ns.Connections {
			node.connections[l] = make([]int32, 0, len(layer))
			for _, nbID := range layer {
				if nb, ok := h.nodes[nbID]; ok {
					node.connections[l] = append(node.connections[l], nb.poolIdx)
				}
			}
		}
	}

	if ep, ok := h.nodes[s.EntryPoint]; ok {
		h.entrySlot = ep.poolIdx
	}

	if s.HasSQ {
		h.codebook = &core.SQCodebook{Min: s.SQMin, Scale: s.SQScale, Dim: s.Dim}
		if !s.Normalized && h.metric.Normalizes() {
			// Older files trained SQ on raw vectors; retrain on the normalized ones.
			vecs := make([]core.Vector, 0, len(h.nodes))
			for _, node := range h.nodes {
				vecs = append(vecs, core.Vector{Embeddings: h.pool.Get(node.poolIdx)})
			}
			if cb := core.NewSQCodebook(vecs); cb != nil {
				h.codebook = cb
			}
		}
		h.sqPool = core.NewInt8VectorPool(s.Dim, h.pool.Slots())
		for _, node := range h.nodes {
			h.sqPool.Set(node.poolIdx, h.codebook.Quantize(h.pool.Get(node.poolIdx)))
		}
	}
	if s.HasPQ {
		subDim := s.Dim / s.PQNSubs
		h.pqCodebook = &core.PQCodebook{
			NSubs: s.PQNSubs, NCentroids: s.PQNCentroids,
			SubDim: subDim, Centroids: s.PQCentroids,
		}
		h.pqPool = core.NewPQPool(s.PQNSubs, h.pool.Slots())
		for _, node := range h.nodes {
			h.pqPool.Set(node.poolIdx, h.pqCodebook.Encode(h.pool.Get(node.poolIdx)))
		}
	}
	h.visitedPool.New = func() any { b := make([]byte, 0); return &b }
	return nil
}
