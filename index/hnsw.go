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

// candidate uses int32 pool slots instead of string IDs to eliminate string
// allocations and per-hop map lookups in the hot search path.
type candidate struct {
	slot int32
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

// hnswNode connections store int32 pool slots — not string IDs.
// At 1M nodes with M=24 this saves ~1.5 GB and eliminates 50M string allocs.
type hnswNode struct {
	id          string
	poolIdx     int32
	connections [][]int32 // each entry is a neighbor's pool slot
}

type HNSWIndex struct {
	mu             sync.RWMutex
	dim            int
	M              int
	efConstruction int
	ef             int
	mL             float64
	nodes          map[string]*hnswNode
	slotToNode     []*hnswNode // O(1) reverse lookup: pool slot → node
	entrySlot      int32       // pool slot of the current entry point
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

func (h *HNSWIndex) Train(vectors []core.Vector) {
	h.mu.Lock()
	defer h.mu.Unlock()

	cb := core.NewSQCodebook(vectors)
	if cb == nil {
		return
	}
	h.codebook = cb
	h.sqPool = core.NewInt8VectorPool(h.dim, max(h.pool.Slots(), 64))
	for _, node := range h.nodes {
		h.sqPool.Set(node.poolIdx, cb.Quantize(h.pool.Get(node.poolIdx)))
	}

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

func (h *HNSWIndex) growSlotToNode(slot int32) {
	for int(slot) >= len(h.slotToNode) {
		h.slotToNode = append(h.slotToNode, nil)
	}
}

// searchLayer runs the beam search at a single HNSW layer.
// visited and candidates use int32 slots — no string map lookups per hop.
func (h *HNSWIndex) searchLayer(query []float32, queryInt8 []int8, distTable []float32, eps []candidate, ef, layer int) []candidate {
	visited := make(map[int32]bool, ef*2)

	cands := &minHeap{}
	W := &maxHeap{}

	for _, ep := range eps {
		heap.Push(cands, ep)
		heap.Push(W, ep)
		visited[ep.slot] = true
	}

	for cands.Len() > 0 {
		c := heap.Pop(cands).(candidate)
		f := (*W)[0]
		if c.dist > f.dist {
			break
		}
		node := h.slotToNode[c.slot]
		if layer >= len(node.connections) {
			continue
		}
		for _, nbSlot := range node.connections[layer] {
			if visited[nbSlot] {
				continue
			}
			visited[nbSlot] = true
			var d float32
			switch {
			case distTable != nil:
				d = h.pqCodebook.DistPQ(h.pqPool.Get(nbSlot), distTable)
			case queryInt8 != nil:
				d = h.codebook.DistInt8(queryInt8, h.sqPool.Get(nbSlot))
			default:
				d = h.dist(query, h.pool.Get(nbSlot))
			}
			f = (*W)[0]
			if d < f.dist || W.Len() < ef {
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
		for _, r := range result {
			if h.dist(h.pool.Get(r.slot), h.pool.Get(c.slot)) < c.dist {
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

	q := vector.Embeddings
	// Use SQ (int8) for construction — PQ is too coarse and degrades graph
	// quality. PQ is used only at search time where re-ranking corrects errors.
	var qInt8 []int8
	if h.codebook != nil {
		qInt8 = h.codebook.Quantize(q)
	}

	ep := []candidate{{h.entrySlot, h.dist(q, h.pool.Get(h.entrySlot))}}

	for layer := h.maxLayer; layer > level; layer-- {
		result := h.searchLayer(q, qInt8, nil, ep, 1, layer)
		ep = result[:1]
	}

	for layer := min(level, h.maxLayer); layer >= 0; layer-- {
		mMax := h.M
		if layer == 0 {
			mMax = h.M * 2
		}
		candidates := h.searchLayer(q, qInt8, nil, ep, h.efConstruction, layer)

		// Re-rank with exact float32 distances before neighbor selection.
		if qInt8 != nil {
			for i := range candidates {
				candidates[i].dist = h.dist(q, h.pool.Get(candidates[i].slot))
			}
			sort.Slice(candidates, func(i, j int) bool { return candidates[i].dist < candidates[j].dist })
		}

		neighbors := h.selectNeighborsHeuristic(q, candidates, mMax)

		node.connections[layer] = make([]int32, len(neighbors))
		for i, nb := range neighbors {
			node.connections[layer][i] = nb.slot
		}

		for _, nb := range neighbors {
			nbNode := h.slotToNode[nb.slot]
			if layer >= len(nbNode.connections) {
				continue
			}
			nbNode.connections[layer] = append(nbNode.connections[layer], poolIdx)
			if len(nbNode.connections[layer]) > mMax {
				conns := nbNode.connections[layer]
				pruned := make([]candidate, len(conns))
				for i, s := range conns {
					pruned[i] = candidate{s, h.dist(h.pool.Get(nbNode.poolIdx), h.pool.Get(s))}
				}
				for i := 1; i < len(pruned); i++ {
					for j := i; j > 0 && pruned[j].dist < pruned[j-1].dist; j-- {
						pruned[j], pruned[j-1] = pruned[j-1], pruned[j]
					}
				}
				pruned = pruned[:mMax]
				nbNode.connections[layer] = make([]int32, mMax)
				for i, p := range pruned {
					nbNode.connections[layer][i] = p.slot
				}
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

	q := query.Embeddings
	// Use SQ (int8) for graph traversal — fast and accurate enough for the beam.
	// PQ is too coarse (subDim=4) and causes the beam to miss good paths,
	// hurting recall. Float32 re-rank after the search corrects SQ errors.
	var qInt8 []int8
	if h.codebook != nil {
		qInt8 = h.codebook.Quantize(q)
	}

	ep := []candidate{{h.entrySlot, h.dist(q, h.pool.Get(h.entrySlot))}}

	for layer := h.maxLayer; layer > 0; layer-- {
		result := h.searchLayer(q, qInt8, nil, ep, 1, layer)
		ep = result[:1]
	}

	candidates := h.searchLayer(q, qInt8, nil, ep, max(h.ef, topK), 0)

	// Re-rank with exact float32 distances.
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
			Distance: candidates[i].dist,
		})
	}
	return results
}

// ---- persistence ----

type hnswNodeState struct {
	Embeddings  []float32
	Connections [][]string // serialized as string IDs for readability/compat
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
		// Convert int32 slots → string IDs for serialization.
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
	h.maxLayer = s.MaxLayer
	h.pool = core.NewVectorPool(s.Dim, len(s.Nodes))
	h.nodes = make(map[string]*hnswNode, len(s.Nodes))
	h.slotToNode = make([]*hnswNode, 0, len(s.Nodes))

	// First pass: add all vectors to pool, build nodes without connections.
	for id, ns := range s.Nodes {
		poolIdx := h.pool.Add(ns.Embeddings)
		node := &hnswNode{id: id, poolIdx: poolIdx}
		h.nodes[id] = node
		h.growSlotToNode(poolIdx)
		h.slotToNode[poolIdx] = node
	}

	// Second pass: resolve string IDs → int32 slots for connections.
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
	return nil
}
