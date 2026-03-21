package index

import (
	"container/heap"
	"encoding/gob"
	"os"

	"github.com/fayez/goatdb/core"
)

type FlatIndex struct {
	dim            int
	distanceMetric core.DistanceMetric
	vectors        map[string][]float32
}

func NewFlatIndex(dim int, metric core.DistanceMetric) *FlatIndex {
	return &FlatIndex{
		dim:            dim,
		distanceMetric: metric,
		vectors:        make(map[string][]float32),
	}
}

func (f *FlatIndex) AddVector(id string, vector core.Vector) {
	f.vectors[id] = vector.Embeddings
}

func (f *FlatIndex) GetVector(id string) (core.Vector, bool) {
	emb, ok := f.vectors[id]
	if !ok {
		return core.Vector{}, false
	}
	return core.Vector{Embeddings: emb}, true
}

func (f *FlatIndex) DeleteVector(id string) bool {
	if _, ok := f.vectors[id]; !ok {
		return false
	}
	delete(f.vectors, id)
	return true
}

func (f *FlatIndex) Search(query core.Vector, topK int) []core.SearchResult {
	rh := &resultHeap{}
	heap.Init(rh)
	for id, emb := range f.vectors {
		v := core.Vector{Embeddings: emb}
		dist := query.Distance(&v, f.distanceMetric)
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

type flatState struct {
	Dim            int
	DistanceMetric core.DistanceMetric
	Vectors        map[string][]float32
}

func (f *FlatIndex) Save(path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()
	return gob.NewEncoder(file).Encode(flatState{
		Dim:            f.dim,
		DistanceMetric: f.distanceMetric,
		Vectors:        f.vectors,
	})
}

func (f *FlatIndex) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	var s flatState
	if err := gob.NewDecoder(file).Decode(&s); err != nil {
		return err
	}
	f.dim = s.Dim
	f.distanceMetric = s.DistanceMetric
	f.vectors = s.Vectors
	return nil
}
