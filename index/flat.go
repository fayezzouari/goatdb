package index

import "github.com/fayez/goatdb/core"

type FlatIndex struct {
	Vectors        map[string]*core.Vector
	DistanceMetric core.DistanceMetric
	Dim            int
}

func (f *FlatIndex) AddVector(collectionName string, id string, vector *core.Vector) {
	if f.Dim != len(vector.Embeddings) {
		panic("Vector dimension does not match index dimension")
	}
	f.Vectors[id] = vector
}

func (f *FlatIndex) GetVector(collectionName string, id string) (core.Vector, bool) {
	vector, exists := f.Vectors[id]
	if !exists {
		return core.Vector{}, false
	}
	return *vector, exists
}

func (f *FlatIndex) DeleteVector(collectionName string, id string) bool {
	if _, exists := f.Vectors[id]; !exists {
		return false
	}
	delete(f.Vectors, id)
	return true
}
