package index

import (
	"container/heap"
	"encoding/gob"
	"math"
	"math/rand"
	"os"
	"runtime"
	"sync"

	"github.com/fayez/goatdb/core"
)

// ivfList stores one inverted list as parallel slices — contiguous iteration,
// no per-entry map overhead.
type ivfList struct {
	slots []int32
	ids   []string
}

// IVFIndex is an Inverted File index backed by a contiguous VectorPool.
// Centroids are stored flat ([nClusters * dim]) for cache-friendly scanning.
// Cosine collections store L2-normalized vectors, so k-means and probing run
// on the unit sphere.
type IVFIndex struct {
	mu             sync.RWMutex
	dim            int
	nClusters      int
	nProbe         int
	centroids      []float32 // [nClusters * dim]
	lists          []ivfList
	idToCluster    map[string]int // O(1) lookup for delete/get
	pool           *core.VectorPool
	distanceMetric core.DistanceMetric
	metric         core.Metric
	trained        bool
}

func NewIVFIndex(dim, nClusters, nProbe int, metric core.DistanceMetric) *IVFIndex {
	return &IVFIndex{
		dim:            dim,
		nClusters:      nClusters,
		nProbe:         nProbe,
		lists:          make([]ivfList, nClusters),
		idToCluster:    make(map[string]int),
		pool:           core.NewVectorPool(dim, 64),
		distanceMetric: metric,
		metric:         core.ResolveMetric(metric),
	}
}

// l2sq returns the squared euclidean distance, using SIMD when available.
// Used for k-means centroid assignment (metric-agnostic, no sqrt needed).
func l2sq(a, b []float32) float32 { return core.L2SqSlices(a, b) }

// centroid returns the flat slice for cluster c.
func (idx *IVFIndex) centroid(c int) []float32 {
	return idx.centroids[c*idx.dim : (c+1)*idx.dim]
}

// nearestCentroid returns the index of the nearest centroid to v (l2sq).
func (idx *IVFIndex) nearestCentroid(v []float32) int {
	best, bestDist := 0, float32(math.MaxFloat32)
	for c := 0; c < idx.nClusters; c++ {
		if d := l2sq(v, idx.centroid(c)); d < bestDist {
			bestDist, best = d, c
		}
	}
	return best
}

func kmeansppInit(vectors []core.Vector, k int) []float32 {
	dim := len(vectors[0].Embeddings)
	centroids := make([]float32, k*dim)

	// First centroid: random data point.
	i := rand.Intn(len(vectors))
	copy(centroids, vectors[i].Embeddings)

	minDists := make([]float32, len(vectors))
	for j := range minDists {
		minDists[j] = math.MaxFloat32
	}

	for ci := 1; ci < k; ci++ {
		prev := centroids[(ci-1)*dim : ci*dim]
		var total float64
		for j, v := range vectors {
			d := l2sq(v.Embeddings, prev)
			if d < minDists[j] {
				minDists[j] = d
			}
			total += float64(minDists[j])
		}
		target := rand.Float64() * total
		cum := 0.0
		chosen := len(vectors) - 1
		for j, d := range minDists {
			cum += float64(d)
			if cum >= target {
				chosen = j
				break
			}
		}
		copy(centroids[ci*dim:], vectors[chosen].Embeddings)
	}
	return centroids
}

// ivfMaxTrainVecs caps the number of vectors used for IVF k-means training.
// Beyond ~200k, centroid quality improves only marginally while cost grows linearly.
const ivfMaxTrainVecs = 200_000

func (idx *IVFIndex) Train(vectors []core.Vector) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	n := len(vectors)
	if n < idx.nClusters {
		idx.nClusters = n
		idx.lists = make([]ivfList, idx.nClusters)
	}

	// Subsample for large datasets.
	train := vectors
	if len(train) > ivfMaxTrainVecs {
		perm := rand.Perm(len(vectors))[:ivfMaxTrainVecs]
		train = make([]core.Vector, ivfMaxTrainVecs)
		for i, p := range perm {
			train[i] = vectors[p]
		}
	}
	train = prepareVectors(idx.metric, train, 0)

	idx.centroids = kmeansppInit(train, idx.nClusters)

	nt := len(train)
	assignments := make([]int, nt)
	newCentroids := make([]float32, idx.nClusters*idx.dim)
	counts := make([]int, idx.nClusters)

	numWorkers := runtime.NumCPU()
	chunkSize := (nt + numWorkers - 1) / numWorkers

	for iter := 0; iter < 30; iter++ {
		// Assignment step — parallelised over the training subsample.
		var wg sync.WaitGroup
		changed := make([]bool, numWorkers)
		for w := 0; w < numWorkers; w++ {
			wg.Add(1)
			go func(w, start, end int) {
				defer wg.Done()
				for i := start; i < end; i++ {
					c := idx.nearestCentroid(train[i].Embeddings)
					if assignments[i] != c {
						assignments[i] = c
						changed[w] = true
					}
				}
			}(w, w*chunkSize, min(nt, (w+1)*chunkSize))
		}
		wg.Wait()

		anyChanged := false
		for _, ch := range changed {
			if ch {
				anyChanged = true
				break
			}
		}
		if !anyChanged {
			break
		}

		// Update step — single threaded (accumulate then divide).
		for i := range newCentroids {
			newCentroids[i] = 0
		}
		for i := range counts {
			counts[i] = 0
		}
		for i, v := range train {
			c := assignments[i]
			counts[c]++
			base := c * idx.dim
			for j, x := range v.Embeddings {
				newCentroids[base+j] += x
			}
		}
		for c := 0; c < idx.nClusters; c++ {
			if counts[c] == 0 {
				continue
			}
			inv := 1.0 / float32(counts[c])
			base := c * idx.dim
			for j := range idx.dim {
				idx.centroids[base+j] = newCentroids[base+j] * inv
			}
		}
	}
	idx.trained = true
}

func (idx *IVFIndex) AddVector(id string, vector core.Vector) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	slot := idx.pool.Add(vector.Embeddings)
	emb := idx.pool.Get(slot)
	idx.metric.PrepareInPlace(emb)
	c := 0
	if idx.trained {
		c = idx.nearestCentroid(emb)
	}
	idx.lists[c].slots = append(idx.lists[c].slots, slot)
	idx.lists[c].ids = append(idx.lists[c].ids, id)
	idx.idToCluster[id] = c
}

func (idx *IVFIndex) GetVector(id string) (core.Vector, bool) {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	c, ok := idx.idToCluster[id]
	if !ok {
		return core.Vector{}, false
	}
	list := &idx.lists[c]
	for i, lid := range list.ids {
		if lid == id {
			emb := make([]float32, idx.dim)
			copy(emb, idx.pool.Get(list.slots[i]))
			return core.Vector{Embeddings: emb}, true
		}
	}
	return core.Vector{}, false
}

func (idx *IVFIndex) DeleteVector(id string) bool {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	c, ok := idx.idToCluster[id]
	if !ok {
		return false
	}
	list := &idx.lists[c]
	for i, lid := range list.ids {
		if lid == id {
			idx.pool.Free(list.slots[i])
			last := len(list.ids) - 1
			list.slots[i] = list.slots[last]
			list.ids[i] = list.ids[last]
			list.slots = list.slots[:last]
			list.ids = list.ids[:last]
			delete(idx.idToCluster, id)
			return true
		}
	}
	return false
}

// topNCentroids returns the indices of the nProbe nearest centroids using a
// partial-sort heap — O(nClusters log nProbe) instead of O(nClusters²).
func (idx *IVFIndex) topNCentroids(query []float32) []int {
	type cd struct {
		i    int
		dist float32
	}
	// Max-heap capped at nProbe entries.
	type cdHeap []cd
	h := make(cdHeap, 0, idx.nProbe)
	hLess := func(a, b cd) bool { return a.dist > b.dist } // max at top
	_ = hLess

	for c := 0; c < idx.nClusters; c++ {
		d := l2sq(query, idx.centroid(c))
		if len(h) < idx.nProbe {
			h = append(h, cd{c, d})
			// sift up
			for i := len(h) - 1; i > 0; {
				p := (i - 1) / 2
				if h[p].dist < h[i].dist {
					h[p], h[i] = h[i], h[p]
					i = p
				} else {
					break
				}
			}
		} else if d < h[0].dist {
			h[0] = cd{c, d}
			// sift down
			n := len(h)
			i := 0
			for {
				l, r, largest := 2*i+1, 2*i+2, i
				if l < n && h[l].dist > h[largest].dist {
					largest = l
				}
				if r < n && h[r].dist > h[largest].dist {
					largest = r
				}
				if largest == i {
					break
				}
				h[i], h[largest] = h[largest], h[i]
				i = largest
			}
		}
	}
	out := make([]int, len(h))
	for i, e := range h {
		out[i] = e.i
	}
	return out
}

func (idx *IVFIndex) Search(query core.Vector, topK int) []core.SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	q := idx.metric.Prepare(query.Embeddings)
	clusterIdxs := idx.topNCentroids(q)

	distFn := idx.metric.Dist
	rh := &resultHeap{}
	heap.Init(rh)
	for _, c := range clusterIdxs {
		list := &idx.lists[c]
		for i, slot := range list.slots {
			dist := distFn(q, idx.pool.Get(slot))
			result := core.SearchResult{Id: list.ids[i], Distance: dist}
			if rh.Len() < topK {
				heap.Push(rh, result)
			} else if dist < (*rh)[0].Distance {
				heap.Pop(rh)
				heap.Push(rh, result)
			}
		}
	}
	for i := range *rh {
		(*rh)[i].Distance = idx.metric.Finalize((*rh)[i].Distance)
	}
	return []core.SearchResult(*rh)
}

// ---- persistence ----

type ivfListState struct {
	Embeddings [][]float32
	IDs        []string
}

type ivfState struct {
	Dim            int
	NClusters      int
	NProbe         int
	DistanceMetric core.DistanceMetric
	Centroids      []float32
	Trained        bool
	Lists          []ivfListState
	Normalized     bool
}

func (idx *IVFIndex) Save(path string) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	lists := make([]ivfListState, len(idx.lists))
	for c, list := range idx.lists {
		embs := make([][]float32, len(list.slots))
		for i, slot := range list.slots {
			emb := make([]float32, idx.dim)
			copy(emb, idx.pool.Get(slot))
			embs[i] = emb
		}
		lists[c] = ivfListState{Embeddings: embs, IDs: list.ids}
	}
	return gob.NewEncoder(file).Encode(ivfState{
		Dim:            idx.dim,
		NClusters:      idx.nClusters,
		NProbe:         idx.nProbe,
		DistanceMetric: idx.distanceMetric,
		Centroids:      idx.centroids,
		Trained:        idx.trained,
		Lists:          lists,
		Normalized:     idx.metric.Normalizes(),
	})
}

func (idx *IVFIndex) Load(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	var s ivfState
	if err := gob.NewDecoder(file).Decode(&s); err != nil {
		return err
	}
	idx.dim = s.Dim
	idx.nClusters = s.NClusters
	idx.nProbe = s.NProbe
	idx.distanceMetric = s.DistanceMetric
	idx.metric = core.ResolveMetric(s.DistanceMetric)
	idx.centroids = s.Centroids
	idx.trained = s.Trained
	if !s.Normalized && idx.metric.Normalizes() {
		// Older files hold raw vectors and centroids; move them onto the unit sphere.
		for c := 0; c < len(idx.centroids)/idx.dim; c++ {
			core.Normalize(idx.centroid(c))
		}
	}

	total := 0
	for _, l := range s.Lists {
		total += len(l.IDs)
	}
	idx.pool = core.NewVectorPool(s.Dim, max(total, 64))
	idx.lists = make([]ivfList, len(s.Lists))
	idx.idToCluster = make(map[string]int, total)

	for c, ls := range s.Lists {
		idx.lists[c].slots = make([]int32, len(ls.IDs))
		idx.lists[c].ids = make([]string, len(ls.IDs))
		for i, id := range ls.IDs {
			slot := idx.pool.Add(ls.Embeddings[i])
			if !s.Normalized {
				idx.metric.PrepareInPlace(idx.pool.Get(slot))
			}
			idx.lists[c].slots[i] = slot
			idx.lists[c].ids[i] = id
			idx.idToCluster[id] = c
		}
	}
	return nil
}
