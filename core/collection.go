package core

import "container/heap"

type DistanceMetric string

const (
	Cosine     DistanceMetric = "cosine"
	Euclidean  DistanceMetric = "euclidean"
	DotProduct DistanceMetric = "dot_product"
	Manhattan  DistanceMetric = "manhattan"
)

type SearchResult struct {
	id       string
	Distance float32
	vector   Vector
}

type resultHeap []SearchResult

type Collection struct {
	id             int
	name           string
	index          string
	vectors        map[string]*Vector
	distanceMetric DistanceMetric
	dim            int
}

func (h resultHeap) Len() int           { return len(h) }
func (h resultHeap) Less(i, j int) bool { return h[i].Distance > h[j].Distance }
func (h resultHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *resultHeap) Push(x any)        { *h = append(*h, x.(SearchResult)) }
func (h *resultHeap) Pop() any          { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }

func (c *Collection) AddVector(id string, vector Vector) {
	if c.dim != len(vector.embeddings) {
		panic("Vector dimension does not match collection dimension")
	}
	c.vectors[id] = &vector
}

func (c *Collection) GetVector(id string) (Vector, bool) {
	vector, exists := c.vectors[id]
	if !exists {
		return Vector{}, false
	}
	return *vector, exists
}

func (c *Collection) DeleteVector(id string) bool {
	if _, exists := c.vectors[id]; !exists {
		return false
	}
	delete(c.vectors, id)
	return true
}

func (c *Collection) Search(query Vector, topK int) []SearchResult {
	if c.dim != len(query.embeddings) {
		panic("Query vector dimension does not match collection dimension")
	}
	h := &resultHeap{}
	heap.Init(h)
	distance := float32(0)

	for id := range c.vectors {
		switch c.distanceMetric {
		case Cosine:
			distance = query.cosine(c.vectors[id])
		case Euclidean:
			distance = query.euclidean(c.vectors[id])
		case Manhattan:
			distance = query.manhattan(c.vectors[id])
		case DotProduct:
			distance = query.dotProduct(c.vectors[id])
		default:
			panic("Unsupported distance metric")
		}
		if h.Len() < topK {
			heap.Push(h, SearchResult{id: id, Distance: distance})
		} else if distance < (*h)[0].Distance {
			heap.Pop(h)
			heap.Push(h, SearchResult{id: id, Distance: distance})
		}

	}
	return []SearchResult(*h)
}
