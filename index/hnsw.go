package index

import (
	"cmp"
	"math"
	"math/rand"
	"slices"
	"sync"

	"github.com/fayezzouari/goatdb/core"
)

type candidate struct {
	slot int32
	dist float32
}

// candMinHeap and candMaxHeap are typed binary heaps over []candidate.
// They avoid container/heap, whose Push(any)/Pop() any box every candidate
// into an interface and allocate on each call.
type candMinHeap []candidate

func (h *candMinHeap) push(c candidate) {
	*h = append(*h, c)
	a := *h
	i := len(a) - 1
	for i > 0 {
		p := (i - 1) / 2
		if a[p].dist <= a[i].dist {
			break
		}
		a[p], a[i] = a[i], a[p]
		i = p
	}
}

func (h *candMinHeap) pop() candidate {
	a := *h
	top := a[0]
	n := len(a) - 1
	a[0] = a[n]
	a = a[:n]
	i := 0
	for {
		l := 2*i + 1
		if l >= n {
			break
		}
		m := l
		if r := l + 1; r < n && a[r].dist < a[l].dist {
			m = r
		}
		if a[i].dist <= a[m].dist {
			break
		}
		a[i], a[m] = a[m], a[i]
		i = m
	}
	*h = a
	return top
}

type candMaxHeap []candidate

func (h *candMaxHeap) push(c candidate) {
	*h = append(*h, c)
	a := *h
	i := len(a) - 1
	for i > 0 {
		p := (i - 1) / 2
		if a[p].dist >= a[i].dist {
			break
		}
		a[p], a[i] = a[i], a[p]
		i = p
	}
}

func (h *candMaxHeap) pop() candidate {
	a := *h
	top := a[0]
	n := len(a) - 1
	a[0] = a[n]
	a = a[:n]
	i := 0
	for {
		l := 2*i + 1
		if l >= n {
			break
		}
		m := l
		if r := l + 1; r < n && a[r].dist > a[l].dist {
			m = r
		}
		if a[i].dist >= a[m].dist {
			break
		}
		a[i], a[m] = a[m], a[i]
		i = m
	}
	*h = a
	return top
}

// searchScratch holds the per-operation buffers used by searchLayer. One
// scratch is taken from HNSWIndex.scratchPool for a whole Search or AddVector

// searchScratch holds the per-operation buffers used by searchLayer. One
// scratch is taken from HNSWIndex.scratchPool for a whole Search or AddVector
// call, so concurrent searches under the read lock never share one.
type searchScratch struct {
	// visited is indexed by slot. Only the entries listed in dirty are
	// non-zero, and they are reset before searchLayer returns.
	visited []byte
	dirty   []int32
	cands   candMinHeap
	W       candMaxHeap
	// result is the buffer searchLayer returns. It is overwritten by the
	// next searchLayer call on the same scratch.
	result []candidate
	ep     []candidate
	// Buffers for neighbor selection in AddVector. selected holds the new
	// node's neighbors while prune* are used to shrink a neighbor's
	// overflowing connection list, so they must not share memory.
	selected    []candidate
	discarded   []candidate
	pruneCands  []candidate
	pruneSel    []candidate
	pruneDiscrd []candidate
}

// HNSWIndex is a Hierarchical Navigable Small World graph.
//
// Nodes are identified by a dense int32 slot. All per-node data lives in
// slot-indexed segmented arrays rather than in per-node heap objects:
//
//   - vecs holds the (prepared) float32 vectors, dim values per slot;
//   - links0 holds the layer-0 adjacency list with a fixed capacity of 2*M,
//     laid out hnswlib-style as [count, link0, ..., link(2M-1)];
//   - upper holds, only for nodes above layer 0, one block of
//     level*(1+M) int32 with the same [count, links...] layout per layer;
//   - levels holds each node's top layer, or -1 for a free slot;
//   - slotIDs maps slot to id, and ids maps id to slot.
//
// Deleted slots are recycled through free.
type HNSWIndex struct {
	mu             sync.RWMutex
	dim            int
	M              int
	efConstruction int
	ef             int
	mL             float64
	// keepPruned (keepPrunedConnections in the paper) fills a node's
	// remaining connection slots with candidates the neighbor heuristic
	// discarded, so sparse regions keep their full degree.
	keepPruned     bool
	distanceMetric core.DistanceMetric
	metric         core.Metric

	ids     map[string]int32
	slotIDs segArray[string]
	vecs    segArray[float32]
	links0  segArray[int32]
	upper   segArray[[]int32]
	levels  segArray[int8]
	// nSlots is the number of slots handed out so far (live or free).
	nSlots int
	free   []int32

	entrySlot int32
	maxLayer  int

	codebook *core.SQCodebook
	// sq holds int8 codes for every slot while codebook is set.
	sq *segArray[int8]
	// trainMu serializes Train calls.
	trainMu sync.Mutex
	// training is set while Train quantizes vectors outside mu. AddVector
	// then records the slots it writes in trainDirty, and Train re-quantizes
	// them with the new codebook when it swaps the codebook in.
	training   bool
	trainDirty []int32
	// scratchPool reuses searchScratch buffers (visited array, heaps, result)
	// across Search and AddVector calls. Flat array lookup is O(1) at ~2ns vs
	// map at ~50ns, and reusing the heaps removes per-push allocations.
	scratchPool sync.Pool
}

// hnswMaxLevel bounds the level drawn for a node so it fits levels' int8.
// With M >= 2 a level above 30 has probability below 2^-30.
const hnswMaxLevel = 100

func NewHNSWIndex(dim, M, efConstruction, ef int, metric core.DistanceMetric) *HNSWIndex {
	h := &HNSWIndex{
		efConstruction: efConstruction,
		ef:             ef,
		keepPruned:     true,
		distanceMetric: metric,
		metric:         core.ResolveMetric(metric),
	}
	h.reset(dim, M, 0)
	h.mL = 1.0 / math.Log(float64(M))
	return h
}

// reset replaces the graph with an empty one sized for about n nodes.
func (h *HNSWIndex) reset(dim, M, n int) {
	h.dim = dim
	h.M = M
	h.ids = make(map[string]int32, n)
	h.slotIDs = newSegArray[string](1)
	h.vecs = newSegArray[float32](dim)
	h.links0 = newSegArray[int32](1 + 2*M)
	h.upper = newSegArray[[]int32](1)
	h.levels = newSegArray[int8](1)
	h.nSlots = 0
	h.free = nil
	h.entrySlot = 0
	h.maxLayer = -1
	h.codebook = nil
	h.sq = nil
}

// trainChunk is the number of vectors Train quantizes per read-lock hold.
const trainChunk = 4096

// testHookTrainChunk, when set by tests, runs after each chunk Train
// quantizes, while no index lock is held.
var testHookTrainChunk func()

// Train builds a scalar quantization codebook from vectors and quantizes
// every indexed vector with it. The codebook is built without any index
// lock, and vectors are quantized in chunks under the read lock, so
// searches keep running and inserts are only delayed by one chunk. The
// write lock is held only to swap the new codebook in.
func (h *HNSWIndex) Train(vectors []core.Vector) {
	h.trainMu.Lock()
	defer h.trainMu.Unlock()

	vectors = prepareVectors(h.metric, vectors, hnswMaxTrainVecs)
	cb := core.NewSQCodebook(vectors)
	if cb == nil {
		return
	}

	h.mu.Lock()
	h.training = true
	h.trainDirty = h.trainDirty[:0]
	n := h.nSlots
	dim := h.dim
	h.mu.Unlock()

	sq := newSegArray[int8](dim)
	sq.grow(n)
	for start := 0; start < n; start += trainChunk {
		end := min(start+trainChunk, n)
		h.mu.RLock()
		if !h.training {
			// Load replaced the index while we were quantizing.
			h.mu.RUnlock()
			return
		}
		for slot := int32(start); slot < int32(end); slot++ {
			if *h.levels.one(slot) >= 0 {
				quantizeInto(cb, sq.at(slot), h.vecs.at(slot))
			}
		}
		h.mu.RUnlock()
		if testHookTrainChunk != nil {
			testHookTrainChunk()
		}
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	if !h.training {
		return
	}
	// Slots written by AddVector after training started may hold vectors
	// quantized from older data (or none at all); redo them.
	sq.grow(h.nSlots)
	for _, slot := range h.trainDirty {
		quantizeInto(cb, sq.at(slot), h.vecs.at(slot))
	}
	h.codebook = cb
	h.sq = &sq
	h.training = false
	h.trainDirty = nil
}

// quantizeInto writes cb's int8 code for v into dst.
func quantizeInto(cb *core.SQCodebook, dst []int8, v []float32) {
	copy(dst, cb.Quantize(v))
}

// hnswMaxTrainVecs caps the normalized copies made for codebook training.
const hnswMaxTrainVecs = 100_000

func (h *HNSWIndex) randomLevel() int {
	return min(int(-math.Log(rand.Float64())*h.mL), hnswMaxLevel)
}

func (h *HNSWIndex) dist(a, b []float32) float32 {
	return h.metric.Dist(a, b)
}

// level returns the top layer of the node at slot, or -1 if slot is free.
func (h *HNSWIndex) level(slot int32) int {
	return int(*h.levels.one(slot))
}

// links returns the [count, links...] block of slot on layer. The node must
// exist on that layer.
func (h *HNSWIndex) links(slot int32, layer int) []int32 {
	if layer == 0 {
		return h.links0.at(slot)
	}
	off := (layer - 1) * (h.M + 1)
	return (*h.upper.one(slot))[off : off+h.M+1 : off+h.M+1]
}

// allocSlot returns a free slot, reusing deleted ones first, with every
// per-slot array grown to cover it.
func (h *HNSWIndex) allocSlot() int32 {
	if n := len(h.free); n > 0 {
		slot := h.free[n-1]
		h.free = h.free[:n-1]
		return slot
	}
	slot := int32(h.nSlots)
	h.nSlots++
	if h.nSlots > h.vecs.len() {
		h.vecs.grow(h.nSlots)
		h.links0.grow(h.nSlots)
		h.upper.grow(h.nSlots)
		h.levels.grow(h.nSlots)
		h.slotIDs.grow(h.nSlots)
	}
	if h.sq != nil && h.nSlots > h.sq.len() {
		h.sq.grow(h.nSlots)
	}
	return slot
}

// initNode fills a freshly allocated slot. The vector is copied and
// prepared for the metric.
func (h *HNSWIndex) initNode(slot int32, id string, emb []float32, level int) {
	v := h.vecs.at(slot)
	copy(v, emb[:h.dim])
	h.metric.PrepareInPlace(v)
	if h.codebook != nil {
		quantizeInto(h.codebook, h.sq.at(slot), v)
	}
	if h.training {
		h.trainDirty = append(h.trainDirty, slot)
	}
	*h.slotIDs.one(slot) = id
	*h.levels.one(slot) = int8(level)
	h.links0.at(slot)[0] = 0
	if level > 0 {
		*h.upper.one(slot) = make([]int32, level*(h.M+1))
	} else {
		*h.upper.one(slot) = nil
	}
	h.ids[id] = slot
}

func (h *HNSWIndex) slotDist(q []float32, qInt8 []int8, slot int32) float32 {
	if qInt8 != nil {
		return h.codebook.DistInt8(qInt8, h.sq.at(slot))
	}
	return h.dist(q, h.vecs.at(slot))
}

func (h *HNSWIndex) getScratch() *searchScratch {
	if s, ok := h.scratchPool.Get().(*searchScratch); ok {
		return s
	}
	return &searchScratch{}
}

func (h *HNSWIndex) putScratch(s *searchScratch) {
	h.scratchPool.Put(s)
}

// searchLayer runs a greedy beam search on one layer and returns up to ef
// candidates sorted by ascending distance.
//
// The returned slice is s.result: it stays valid only until the next
// searchLayer call on the same scratch. eps may alias s.result, because the
// entry points are copied into the heaps before s.result is rewritten.
func (h *HNSWIndex) searchLayer(s *searchScratch, query []float32, queryInt8 []int8, eps []candidate, ef, layer int) []candidate {
	vb := s.visited
	if len(vb) < h.nSlots {
		// Grow with headroom: sizing the buffer to exactly nSlots would
		// reallocate it on every insert.
		vb = make([]byte, h.nSlots+h.nSlots/2+64)
	}
	s.visited = vb
	dirty := s.dirty[:0]
	cands := s.cands[:0]
	W := s.W[:0]

	for _, ep := range eps {
		cands.push(ep)
		W.push(ep)
		if vb[ep.slot] == 0 {
			vb[ep.slot] = 1
			dirty = append(dirty, ep.slot)
		}
	}

	for len(cands) > 0 {
		c := cands.pop()
		if c.dist > W[0].dist {
			break
		}
		if layer > h.level(c.slot) {
			continue
		}
		blk := h.links(c.slot, layer)
		for _, nbSlot := range blk[1 : 1+blk[0]] {
			if vb[nbSlot] != 0 {
				continue
			}
			vb[nbSlot] = 1
			dirty = append(dirty, nbSlot)
			if h.level(nbSlot) < layer {
				// A free slot, or a recycled one, left behind by a delete.
				continue
			}
			d := h.slotDist(query, queryInt8, nbSlot)
			if d < W[0].dist || len(W) < ef {
				cands.push(candidate{nbSlot, d})
				W.push(candidate{nbSlot, d})
				if len(W) > ef {
					W.pop()
				}
			}
		}
	}

	n := len(W)
	result := s.result[:0]
	if cap(result) < n {
		result = make([]candidate, n)
	} else {
		result = result[:n]
	}
	for i := n - 1; i >= 0; i-- {
		result[i] = W.pop()
	}

	for _, slot := range dirty {
		vb[slot] = 0
	}
	s.dirty = dirty[:0]
	s.cands = cands[:0]
	s.W = W[:0]
	s.result = result
	return result
}

// sortCandidates sorts by ascending distance without the reflection and
// closure allocations of sort.Slice.
func sortCandidates(c []candidate) {
	slices.SortFunc(c, func(a, b candidate) int { return cmp.Compare(a.dist, b.dist) })
}

// selectNeighbors picks up to M neighbors from candidates (sorted by
// ascending distance to the base node) with the heuristic of Malkov &
// Yashunin, Algorithm 4: a candidate is kept only if it is closer to the base
// node than to every neighbor selected so far. This spreads links across
// clusters instead of pointing them all into the nearest dense one. With
// keepPruned, discarded candidates fill any remaining slots in
// distance order.
//
// sel and discarded are reusable buffers; the selection is returned in sel's
// backing array and must not alias candidates.
func (h *HNSWIndex) selectNeighbors(candidates []candidate, M int, sel, discarded []candidate) (selOut, discardedOut []candidate) {
	sel = sel[:0]
	discarded = discarded[:0]
	if h.keepPruned && len(candidates) <= M {
		return append(sel, candidates...), discarded
	}
	for _, c := range candidates {
		if len(sel) >= M {
			break
		}
		cVec := h.vecs.at(c.slot)
		good := true
		for _, r := range sel {
			if h.dist(cVec, h.vecs.at(r.slot)) < c.dist {
				good = false
				break
			}
		}
		if good {
			sel = append(sel, c)
		} else {
			discarded = append(discarded, c)
		}
	}
	if h.keepPruned {
		for _, c := range discarded {
			if len(sel) >= M {
				break
			}
			sel = append(sel, c)
		}
	}
	return sel, discarded
}

// addLink adds a link from slot to newSlot on layer. When slot's list is
// full (mMax links), the list plus newSlot is shrunk back to mMax entries
// with the same heuristic as selectNeighbors.
func (h *HNSWIndex) addLink(s *searchScratch, slot int32, layer, mMax int, newSlot int32) {
	blk := h.links(slot, layer)
	n := int(blk[0])
	if n < mMax {
		blk[1+n] = newSlot
		blk[0]++
		return
	}
	vec := h.vecs.at(slot)
	cands := s.pruneCands[:0]
	for _, c := range blk[1 : 1+n] {
		cands = append(cands, candidate{c, h.dist(vec, h.vecs.at(c))})
	}
	cands = append(cands, candidate{newSlot, h.dist(vec, h.vecs.at(newSlot))})
	sortCandidates(cands)
	sel, disc := h.selectNeighbors(cands, mMax, s.pruneSel, s.pruneDiscrd)
	for i, c := range sel {
		blk[1+i] = c.slot
	}
	blk[0] = int32(len(sel))
	s.pruneCands, s.pruneSel, s.pruneDiscrd = cands, sel, disc
}

func (h *HNSWIndex) AddVector(id string, vector core.Vector) {
	level := h.randomLevel()

	h.mu.Lock()
	defer h.mu.Unlock()

	slot := h.allocSlot()
	h.initNode(slot, id, vector.Embeddings, level)
	q := h.vecs.at(slot)

	if h.maxLayer == -1 {
		h.entrySlot = slot
		h.maxLayer = level
		return
	}

	var qInt8 []int8
	if h.codebook != nil {
		qInt8 = h.sq.at(slot)
	}

	s := h.getScratch()
	defer h.putScratch(s)

	ep := append(s.ep[:0], candidate{h.entrySlot, h.slotDist(q, qInt8, h.entrySlot)})
	s.ep = ep

	for layer := h.maxLayer; layer > level; layer-- {
		result := h.searchLayer(s, q, qInt8, ep, 1, layer)
		ep = result[:1]
	}

	for layer := min(level, h.maxLayer); layer >= 0; layer-- {
		mMax := h.M
		if layer == 0 {
			mMax = h.M * 2
		}
		// candidates aliases s.result; it is fully consumed (and reused as
		// the next layer's entry points) before searchLayer runs again.
		candidates := h.searchLayer(s, q, qInt8, ep, h.efConstruction, layer)
		if qInt8 != nil {
			for i := range candidates {
				candidates[i].dist = h.dist(q, h.vecs.at(candidates[i].slot))
			}
			sortCandidates(candidates)
		}
		// A recycled slot can be reached through links left behind when it
		// was deleted; never link a node to itself.
		candidates = slices.DeleteFunc(candidates, func(c candidate) bool { return c.slot == slot })
		// As in the paper (Algorithm 1), the new node links to M neighbors
		// on every layer; back-links can grow its list up to mMax.
		neighbors, disc := h.selectNeighbors(candidates, h.M, s.selected, s.discarded)
		s.selected, s.discarded = neighbors, disc
		blk := h.links(slot, layer)
		for i, nb := range neighbors {
			blk[1+i] = nb.slot
		}
		blk[0] = int32(len(neighbors))
		for _, nb := range neighbors {
			h.addLink(s, nb.slot, layer, mMax, slot)
		}
		ep = candidates
	}

	if level > h.maxLayer {
		h.maxLayer = level
		h.entrySlot = slot
	}
}

func (h *HNSWIndex) GetVector(id string) (core.Vector, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	slot, ok := h.ids[id]
	if !ok {
		return core.Vector{}, false
	}
	emb := make([]float32, h.dim)
	copy(emb, h.vecs.at(slot))
	return core.Vector{Embeddings: emb}, true
}

func (h *HNSWIndex) DeleteVector(id string) bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	slot, exists := h.ids[id]
	if !exists {
		return false
	}
	// Remove the back-links of the node's own neighbors. Links from nodes
	// that are not its neighbors stay behind; searches skip them while the
	// slot is free and treat them as ordinary edges once it is reused.
	for layer := h.level(slot); layer >= 0; layer-- {
		blk := h.links(slot, layer)
		for _, nb := range blk[1 : 1+blk[0]] {
			if nb == slot || h.level(nb) < layer {
				continue
			}
			nbBlk := h.links(nb, layer)
			n := nbBlk[0]
			out := nbBlk[1:1:len(nbBlk)]
			for _, x := range nbBlk[1 : 1+n] {
				if x != slot {
					out = append(out, x)
				}
			}
			nbBlk[0] = int32(len(out))
		}
	}
	*h.levels.one(slot) = -1
	*h.upper.one(slot) = nil
	*h.slotIDs.one(slot) = ""
	h.links0.at(slot)[0] = 0
	h.free = append(h.free, slot)
	delete(h.ids, id)

	if h.entrySlot == slot {
		h.maxLayer = -1
		for _, s := range h.ids {
			if lvl := h.level(s); lvl > h.maxLayer {
				h.maxLayer = lvl
				h.entrySlot = s
			}
		}
	}
	return true
}

func (h *HNSWIndex) Search(query core.Vector, topK int) []core.SearchResult {
	return h.SearchEf(query, topK, 0)
}

// SearchEf searches with beam width max(ef, topK) at layer 0. An ef <= 0 uses
// the index's configured ef.
func (h *HNSWIndex) SearchEf(query core.Vector, topK, ef int) []core.SearchResult {
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

	s := h.getScratch()
	defer h.putScratch(s)

	ep := append(s.ep[:0], candidate{h.entrySlot, h.slotDist(q, qInt8, h.entrySlot)})
	s.ep = ep

	for layer := h.maxLayer; layer > 0; layer-- {
		result := h.searchLayer(s, q, qInt8, ep, 1, layer)
		ep = result[:1]
	}

	if ef <= 0 {
		ef = h.ef
	}
	candidates := h.searchLayer(s, q, qInt8, ep, max(ef, topK), 0)

	if qInt8 != nil {
		for i := range candidates {
			candidates[i].dist = h.dist(q, h.vecs.at(candidates[i].slot))
		}
		sortCandidates(candidates)
	}

	results := make([]core.SearchResult, 0, topK)
	for i := 0; i < topK && i < len(candidates); i++ {
		results = append(results, core.SearchResult{
			Id:       *h.slotIDs.one(candidates[i].slot),
			Distance: h.metric.Finalize(candidates[i].dist),
		})
	}
	return results
}
