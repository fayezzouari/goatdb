package core

type DistanceMetruc string

const (
	Cosine     DistanceMetruc = "cosine"
	Euclidean  DistanceMetruc = "euclidean"
	DotProduct DistanceMetruc = "dot_product"
	Manhattan  DistanceMetruc = "manhattan"
)

type Collection struct {
	id             int
	name           string
	index          string
	vectors        map[string]Vector
	distanceMetric string
	dim            int
}

func (c *Collection) AddVector(id string, vector Vector) {
	if c.dim != len(vector.embeddings) {
		panic("Vector dimension does not match collection dimension")
	}
	c.vectors[id] = vector
}

func (c *Collection) GetVector(id string) (Vector, bool) {
	vector, exists := c.vectors[id]
	if !exists {
		return Vector{}, false
	}
	return vector, exists
}

func (c *Collection) DeleteVector(id string) bool {
	if _, exists := c.vectors[id]; !exists {
		return false
	}
	delete(c.vectors, id)
	return true
}
