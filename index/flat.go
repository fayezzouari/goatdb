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
	pool           *core.VectorPool
	idToSlot       map[string]int32
	slotToID       []string
}

func NewFlatIndex(dim int, metric core.DistanceMetric) *FlatIndex {
	return &FlatIndex{
		dim:            dim,
		distanceMetric: metric,
		pool:           core.NewVectorPool(dim, 64),
		idToSlot:       make(map[string]int32),
	}
}

func (f *FlatIndex) AddVector(id string, vector core.Vector) {
	slot := f.pool.Add(vector.Embeddings)
	f.idToSlot[id] = slot
	for int(slot) >= len(f.slotToID) {
		f.slotToID = append(f.slotToID, "")
	}
	f.slotToID[slot] = id
}

func (f *FlatIndex) GetVector(id string) (core.Vector, bool) {
	slot, ok := f.idToSlot[id]
	if !ok {
		return core.Vector{}, false
	}
	emb := make([]float32, f.dim)
	copy(emb, f.pool.Get(slot))
	return core.Vector{Embeddings: emb}, true
}

func (f *FlatIndex) DeleteVector(id string) bool {
	slot, ok := f.idToSlot[id]
	if !ok {
		return false
	}
	f.pool.Free(slot)
	f.slotToID[slot] = ""
	delete(f.idToSlot, id)
	return true
}

func (f *FlatIndex) Search(query core.Vector, topK int) []core.SearchResult {
	q := query.Embeddings
	rh := &resultHeap{}
	heap.Init(rh)
	f.pool.ForEach(func(slot int32, emb []float32) {
		dist := core.DistSlices(q, emb, f.distanceMetric)
		result := core.SearchResult{Id: f.slotToID[slot], Distance: dist}
		if rh.Len() < topK {
			heap.Push(rh, result)
		} else if dist < (*rh)[0].Distance {
			heap.Pop(rh)
			heap.Push(rh, result)
		}
	})
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
	vectors := make(map[string][]float32, len(f.idToSlot))
	for id, slot := range f.idToSlot {
		emb := make([]float32, f.dim)
		copy(emb, f.pool.Get(slot))
		vectors[id] = emb
	}
	return gob.NewEncoder(file).Encode(flatState{
		Dim:            f.dim,
		DistanceMetric: f.distanceMetric,
		Vectors:        vectors,
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
	f.pool = core.NewVectorPool(s.Dim, len(s.Vectors))
	f.idToSlot = make(map[string]int32, len(s.Vectors))
	f.slotToID = make([]string, 0, len(s.Vectors))
	for id, emb := range s.Vectors {
		slot := f.pool.Add(emb)
		f.idToSlot[id] = slot
		for int(slot) >= len(f.slotToID) {
			f.slotToID = append(f.slotToID, "")
		}
		f.slotToID[slot] = id
	}
	return nil
}
