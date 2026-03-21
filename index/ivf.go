package index

import (
	"container/heap"
	"encoding/gob"
	"math"
	"math/rand"
	"os"

	"github.com/fayez/goatdb/core"
)

type IVFIndex struct {
	dim            int
	nClusters      int
	nProbe         int
	centroids      [][]float32
	lists          []map[string]core.Vector
	distanceMetric core.DistanceMetric
	trained        bool
}

func NewIVFIndex(dim, nClusters, nProbe int, metric core.DistanceMetric) *IVFIndex {
	lists := make([]map[string]core.Vector, nClusters)
	for i := range lists {
		lists[i] = make(map[string]core.Vector)
	}
	return &IVFIndex{
		dim:            dim,
		nClusters:      nClusters,
		nProbe:         nProbe,
		lists:          lists,
		distanceMetric: metric,
	}
}

func (idx *IVFIndex) Train(vectors []core.Vector) {
	n := len(vectors)
	if n < idx.nClusters {
		idx.nClusters = n
	}

	perm := rand.Perm(n)
	idx.centroids = make([][]float32, idx.nClusters)
	for i := range idx.centroids {
		c := make([]float32, idx.dim)
		copy(c, vectors[perm[i]].Embeddings)
		idx.centroids[i] = c
	}

	for iter := 0; iter < 20; iter++ {
		sums := make([][]float32, idx.nClusters)
		counts := make([]int, idx.nClusters)
		for i := range sums {
			sums[i] = make([]float32, idx.dim)
		}
		for _, v := range vectors {
			c := idx.nearestCentroid(v.Embeddings)
			counts[c]++
			for j, val := range v.Embeddings {
				sums[c][j] += val
			}
		}
		for i := range idx.centroids {
			if counts[i] == 0 {
				continue
			}
			for j := range idx.centroids[i] {
				idx.centroids[i][j] = sums[i][j] / float32(counts[i])
			}
		}
	}
	idx.trained = true
}

func (idx *IVFIndex) nearestCentroid(v []float32) int {
	best, bestDist := 0, float32(math.MaxFloat32)
	for i, c := range idx.centroids {
		d := l2sq(v, c)
		if d < bestDist {
			bestDist = d
			best = i
		}
	}
	return best
}

func l2sq(a, b []float32) float32 {
	var d float32
	for i := range a {
		diff := a[i] - b[i]
		d += diff * diff
	}
	return d
}

func (idx *IVFIndex) AddVector(_ string, id string, vector core.Vector) {
	if !idx.trained {
		panic("IVFIndex must be trained before adding vectors")
	}
	c := idx.nearestCentroid(vector.Embeddings)
	idx.lists[c][id] = vector
}

func (idx *IVFIndex) GetVector(_ string, id string) (core.Vector, bool) {
	for _, list := range idx.lists {
		if v, ok := list[id]; ok {
			return v, true
		}
	}
	return core.Vector{}, false
}

func (idx *IVFIndex) DeleteVector(_ string, id string) bool {
	for _, list := range idx.lists {
		if _, ok := list[id]; ok {
			delete(list, id)
			return true
		}
	}
	return false
}

func (idx *IVFIndex) Search(_ string, query core.Vector, topK int) []core.SearchResult {
	type centDist struct {
		i    int
		dist float32
	}
	dists := make([]centDist, len(idx.centroids))
	for i, c := range idx.centroids {
		dists[i] = centDist{i, l2sq(query.Embeddings, c)}
	}

	nProbe := min(idx.nProbe, len(dists))
	for i := 0; i < nProbe; i++ {
		for j := i + 1; j < len(dists); j++ {
			if dists[j].dist < dists[i].dist {
				dists[i], dists[j] = dists[j], dists[i]
			}
		}
	}

	rh := &resultHeap{}
	heap.Init(rh)
	for i := 0; i < nProbe; i++ {
		for id, v := range idx.lists[dists[i].i] {
			vCopy := v
			dist := query.Distance(&vCopy, idx.distanceMetric)
			result := core.SearchResult{Id: id, Distance: dist}
			if rh.Len() < topK {
				heap.Push(rh, result)
			} else if dist < (*rh)[0].Distance {
				heap.Pop(rh)
				heap.Push(rh, result)
			}
		}
	}
	return []core.SearchResult(*rh)
}

type ivfState struct {
	Dim            int
	NClusters      int
	NProbe         int
	DistanceMetric core.DistanceMetric
	Centroids      [][]float32
	Trained        bool
	Lists          []map[string][]float32
}

func (idx *IVFIndex) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	lists := make([]map[string][]float32, len(idx.lists))
	for i, list := range idx.lists {
		lists[i] = make(map[string][]float32, len(list))
		for id, v := range list {
			lists[i][id] = v.Embeddings
		}
	}
	return gob.NewEncoder(file).Encode(ivfState{
		Dim:            idx.dim,
		NClusters:      idx.nClusters,
		NProbe:         idx.nProbe,
		DistanceMetric: idx.distanceMetric,
		Centroids:      idx.centroids,
		Trained:        idx.trained,
		Lists:          lists,
	})
}

func (idx *IVFIndex) Load(path string) error {
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
	idx.centroids = s.Centroids
	idx.trained = s.Trained
	idx.lists = make([]map[string]core.Vector, len(s.Lists))
	for i, list := range s.Lists {
		idx.lists[i] = make(map[string]core.Vector, len(list))
		for id, emb := range list {
			idx.lists[i][id] = core.Vector{Embeddings: emb}
		}
	}
	return nil
}
