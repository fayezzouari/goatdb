package index

import (
	"cmp"
	"math"
	"math/rand"
	"slices"
	"sync"
	"sync/atomic"

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
// call, so concurrent operations never share one.
type searchScratch struct {
	// visited is indexed by slot. Only the entries listed in dirty are
	// non-zero, and they are reset before searchLayer returns.
	visited []byte
	dirty   []int32
	cands   candMinHeap
	W       candMaxHeap
	// nbs receives a copy of a node's links, taken under the node's lock.
	nbs []int32
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
// # Concurrency
//
// Searches and inserts run in parallel, as in hnswlib. They hold mu for
// reading; deletes, Load and the codebook swap of Train hold it for writing.
// Under the read lock:
//
//   - each node's link lists are guarded by its entry in locks; searches
//     copy a list under the lock and compute distances outside it;
//   - allocMu guards slot allocation and the growth of the slot arrays,
//     which never moves existing slots;
//   - entry holds the entry point and the top layer, updated under epMu;
//   - idsMu guards ids;
//   - a node's vector, codes, id and level are written once, under its
//     lock, before any link to it is published, and stay unchanged while
//     it is live, so they are read without locks.
//
// A deleted slot can still be reached through links left behind by the
// delete, so an insert that recycles a free slot runs under the write lock.
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

	idsMu   sync.RWMutex
	ids     map[string]int32
	slotIDs *segArray[string]
	vecs    *segArray[float32]
	links0  *segArray[int32]
	upper   *segArray[[]int32]
	levels  *segArray[int8]
	locks   *segArray[sync.Mutex]

	allocMu sync.Mutex
	// nSlots is the number of slots handed out so far (live or free).
	nSlots atomic.Int32
	free   []int32

	// entry packs the entry point slot (low 32 bits) and maxLayer+1 (high
	// 32 bits); zero means an empty graph. Updates hold epMu.
	epMu  sync.Mutex
	entry atomic.Uint64

	codebook *core.SQCodebook
	// sq holds int8 codes for every slot while codebook is set.
	sq *segArray[int8]
	// trainMu serializes Train calls.
	trainMu sync.Mutex
	// training is set while Train quantizes vectors outside the write lock.
	// AddVector then records the slots it writes in trainDirty, and Train
	// re-quantizes them with the new codebook when it swaps the codebook in.
	training   bool
	dirtyMu    sync.Mutex
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

// reset replaces the graph with an empty one sized for about n nodes. The
// caller must have exclusive access to h.
func (h *HNSWIndex) reset(dim, M, n int) {
	h.dim = dim
	h.M = M
	h.ids = make(map[string]int32, n)
	h.slotIDs = newSegArray[string](1)
	h.vecs = newSegArray[float32](dim)
	h.links0 = newSegArray[int32](1 + 2*M)
	h.upper = newSegArray[[]int32](1)
	h.levels = newSegArray[int8](1)
	h.locks = newSegArray[sync.Mutex](1)
	h.nSlots.Store(0)
	h.free = nil
	h.entry.Store(0)
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
// searches and inserts keep running. The write lock is held only to swap
// the new codebook in.
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
	n := int(h.nSlots.Load())
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
			// The slot may be one an insert is filling right now; its
			// lock orders the two. A slot filled after this read is in
			// trainDirty.
			l := h.locks.one(slot)
			l.Lock()
			if h.level(slot) >= 0 {
				quantizeInto(cb, sq.at(slot), h.vecs.at(slot))
			}
			l.Unlock()
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
	sq.grow(int(h.nSlots.Load()))
	for _, slot := range h.trainDirty {
		quantizeInto(cb, sq.at(slot), h.vecs.at(slot))
	}
	h.codebook = cb
	h.sq = sq
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
// exist on that layer. Under the read lock, the block is guarded by the
// node's lock.
func (h *HNSWIndex) links(slot int32, layer int) []int32 {
	if layer == 0 {
		return h.links0.at(slot)
	}
	off := (layer - 1) * (h.M + 1)
	return (*h.upper.one(slot))[off : off+h.M+1 : off+h.M+1]
}

// entryPoint returns the entry point and the top layer, which is -1 for an
// empty graph.
func (h *HNSWIndex) entryPoint() (slot int32, maxLayer int) {
	e := h.entry.Load()
	return int32(uint32(e)), int(e>>32) - 1
}

func (h *HNSWIndex) setEntryPoint(slot int32, maxLayer int) {
	h.entry.Store(uint64(maxLayer+1)<<32 | uint64(uint32(slot)))
}

// copyLinks appends slot's links on layer to dst under the node's lock.
func (h *HNSWIndex) copyLinks(dst []int32, slot int32, layer int) []int32 {
	l := h.locks.one(slot)
	l.Lock()
	blk := h.links(slot, layer)
	dst = append(dst, blk[1:1+blk[0]]...)
	l.Unlock()
	return dst
}

// allocSlot returns a slot with every per-slot array grown to cover it. With
// reuse it returns a free slot if there is one, which requires the write
// lock.
func (h *HNSWIndex) allocSlot(reuse bool) int32 {
	h.allocMu.Lock()
	defer h.allocMu.Unlock()
	if n := len(h.free); reuse && n > 0 {
		slot := h.free[n-1]
		h.free = h.free[:n-1]
		return slot
	}
	slot := h.nSlots.Load()
	n := int(slot) + 1
	if n > h.vecs.len() {
		h.vecs.grow(n)
		h.links0.grow(n)
		h.upper.grow(n)
		h.levels.grow(n)
		h.slotIDs.grow(n)
		h.locks.grow(n)
	}
	if h.sq != nil && n > h.sq.len() {
		h.sq.grow(n)
	}
	h.nSlots.Store(int32(n))
	return slot
}

func (h *HNSWIndex) hasFree() bool {
	h.allocMu.Lock()
	defer h.allocMu.Unlock()
	return len(h.free) > 0
}

// initNode fills a newly allocated slot, which no other node links to yet
// (or, for a recycled slot, while the write lock is held). The vector is
// copied and prepared for the metric. The id is published last, so
// GetVector never sees a half-written vector.
func (h *HNSWIndex) initNode(slot int32, id string, emb []float32, level int) {
	l := h.locks.one(slot)
	l.Lock()
	v := h.vecs.at(slot)
	copy(v, emb[:h.dim])
	h.metric.PrepareInPlace(v)
	if h.codebook != nil {
		quantizeInto(h.codebook, h.sq.at(slot), v)
	}
	*h.slotIDs.one(slot) = id
	*h.levels.one(slot) = int8(level)
	h.links0.at(slot)[0] = 0
	if level > 0 {
		*h.upper.one(slot) = make([]int32, level*(h.M+1))
	} else {
		*h.upper.one(slot) = nil
	}
	l.Unlock()

	if h.training {
		h.dirtyMu.Lock()
		h.trainDirty = append(h.trainDirty, slot)
		h.dirtyMu.Unlock()
	}
	h.idsMu.Lock()
	h.ids[id] = slot
	h.idsMu.Unlock()
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

// growVisited returns vb grown to cover slot, keeping its marks.
func growVisited(vb []byte, slot int32) []byte {
	n := int(slot) + 1
	nb := make([]byte, n+n/2+64)
	copy(nb, vb)
	return nb
}

// searchLayer runs a greedy beam search on one layer and returns up to ef
// candidates sorted by ascending distance.
//
// The returned slice is s.result: it stays valid only until the next
// searchLayer call on the same scratch. eps may alias s.result, because the
// entry points are copied into the heaps before s.result is rewritten.
func (h *HNSWIndex) searchLayer(s *searchScratch, query []float32, queryInt8 []int8, eps []candidate, ef, layer int) []candidate {
	vb := s.visited
	if n := int(h.nSlots.Load()); len(vb) < n {
		// Grow with headroom: sizing the buffer to exactly nSlots would
		// reallocate it on every insert.
		vb = make([]byte, n+n/2+64)
	}
	dirty := s.dirty[:0]
	cands := s.cands[:0]
	W := s.W[:0]

	for _, ep := range eps {
		cands.push(ep)
		W.push(ep)
		if int(ep.slot) >= len(vb) {
			vb = growVisited(vb, ep.slot)
		}
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
		s.nbs = h.copyLinks(s.nbs[:0], c.slot, layer)
		for _, nbSlot := range s.nbs {
			if int(nbSlot) >= len(vb) {
				// Allocated by an insert that started after this search.
				vb = growVisited(vb, nbSlot)
			}
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
	s.visited = vb
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

// addLink adds a link from slot to newSlot on layer, under slot's lock.
// When slot's list is full (mMax links), the list plus newSlot is shrunk
// back to mMax entries with the same heuristic as selectNeighbors.
func (h *HNSWIndex) addLink(s *searchScratch, slot int32, layer, mMax int, newSlot int32) {
	l := h.locks.one(slot)
	l.Lock()
	defer l.Unlock()
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

// AddVector inserts a vector. It is safe to call concurrently with itself
// and with every other method.
func (h *HNSWIndex) AddVector(id string, vector core.Vector) {
	level := h.randomLevel()

	// Recycling a free slot needs the write lock (see HNSWIndex); fresh
	// slots are filled under the read lock, in parallel with other calls.
	if h.hasFree() {
		h.mu.Lock()
		defer h.mu.Unlock()
		h.insert(h.allocSlot(true), id, vector.Embeddings, level)
		return
	}
	h.mu.RLock()
	defer h.mu.RUnlock()
	h.insert(h.allocSlot(false), id, vector.Embeddings, level)
}

func (h *HNSWIndex) insert(slot int32, id string, emb []float32, level int) {
	h.initNode(slot, id, emb, level)
	q := h.vecs.at(slot)

	h.epMu.Lock()
	entry, maxLayer := h.entryPoint()
	if maxLayer == -1 {
		h.setEntryPoint(slot, level)
		h.epMu.Unlock()
		return
	}
	h.epMu.Unlock()

	var qInt8 []int8
	if h.codebook != nil {
		qInt8 = h.sq.at(slot)
	}

	s := h.getScratch()
	defer h.putScratch(s)

	ep := append(s.ep[:0], candidate{entry, h.slotDist(q, qInt8, entry)})
	s.ep = ep

	for layer := maxLayer; layer > level; layer-- {
		result := h.searchLayer(s, q, qInt8, ep, 1, layer)
		ep = result[:1]
	}

	top := min(level, maxLayer)
	for layer := top; layer >= 0; layer-- {
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
		l := h.locks.one(slot)
		l.Lock()
		blk := h.links(slot, layer)
		for i, nb := range neighbors {
			blk[1+i] = nb.slot
		}
		blk[0] = int32(len(neighbors))
		l.Unlock()
		ep = candidates
		if len(ep) == 0 {
			break
		}
	}

	// Publish the node bottom-up: once another search can reach it on a
	// layer, its links on every layer below are already in place, so a
	// search descending through it never lands on an unlinked node. Until
	// its back-links on a layer exist nobody else links to it there, so its
	// own list on that layer is still exactly the selection above.
	for layer := 0; layer <= top; layer++ {
		mMax := h.M
		if layer == 0 {
			mMax = h.M * 2
		}
		s.nbs = h.copyLinks(s.nbs[:0], slot, layer)
		for _, nb := range s.nbs {
			h.addLink(s, nb, layer, mMax, slot)
		}
	}

	if level > maxLayer {
		h.epMu.Lock()
		if _, cur := h.entryPoint(); level > cur {
			h.setEntryPoint(slot, level)
		}
		h.epMu.Unlock()
	}
}

func (h *HNSWIndex) GetVector(id string) (core.Vector, bool) {
	h.mu.RLock()
	defer h.mu.RUnlock()
	h.idsMu.RLock()
	slot, ok := h.ids[id]
	h.idsMu.RUnlock()
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
	h.allocMu.Lock()
	h.free = append(h.free, slot)
	h.allocMu.Unlock()
	delete(h.ids, id)

	if entry, _ := h.entryPoint(); entry == slot {
		newEntry, maxLayer := int32(0), -1
		for _, s := range h.ids {
			if lvl := h.level(s); lvl > maxLayer {
				newEntry, maxLayer = s, lvl
			}
		}
		h.setEntryPoint(newEntry, maxLayer)
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
	entry, maxLayer := h.entryPoint()
	if maxLayer == -1 {
		return nil
	}

	q := h.metric.Prepare(query.Embeddings)
	var qInt8 []int8
	if h.codebook != nil {
		qInt8 = h.codebook.Quantize(q)
	}

	s := h.getScratch()
	defer h.putScratch(s)

	ep := append(s.ep[:0], candidate{entry, h.slotDist(q, qInt8, entry)})
	s.ep = ep

	for layer := maxLayer; layer > 0; layer-- {
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

// ConcurrentAdd reports that AddVector calls may run in parallel and scale
// with the number of cores.
func (h *HNSWIndex) ConcurrentAdd() bool { return true }
