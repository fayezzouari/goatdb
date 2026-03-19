package core

type DistanceMetruc string

const (
	Cosine     DistanceMetruc = "cosine"
	Euclidean  DistanceMetruc = "euclidean"
	DotProduct DistanceMetruc = "dot_product"
	Manhattan  DistanceMetruc = "manhattan"
)

type SearchResult struct {
	id       string
	Distance float32
	vector   Vector
}

type Collection struct {
	id             int
	name           string
	index          string
	vectors        map[string]*Vector
	distanceMetric DistanceMetruc
	dim            int
}

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
	results := make([]SearchResult, 0, len(c.vectors))
	for id := range c.vectors {
		var distance float32
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
		results = append(results, SearchResult{id: id, Distance: distance})
	}
	return results
}
