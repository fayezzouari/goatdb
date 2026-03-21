package core

type DistanceMetric string

const (
	Cosine     DistanceMetric = "cosine"
	Euclidean  DistanceMetric = "euclidean"
	DotProduct DistanceMetric = "dot_product"
	Manhattan  DistanceMetric = "manhattan"
)

type SearchResult struct {
	Id       string
	Distance float32
	Vector   Vector
}

type Collection struct {
	id   int
	name string
	dim  int
	index Index
}

func (c *Collection) AddVector(id string, vector Vector) {
	if c.dim != len(vector.Embeddings) {
		panic("Vector dimension does not match collection dimension")
	}
	c.index.AddVector(c.name, id, vector)
}

func (c *Collection) GetVector(id string) (Vector, bool) {
	return c.index.GetVector(c.name, id)
}

func (c *Collection) DeleteVector(id string) bool {
	return c.index.DeleteVector(c.name, id)
}

func (c *Collection) Search(query Vector, topK int) []SearchResult {
	if c.dim != len(query.Embeddings) {
		panic("Query vector dimension does not match collection dimension")
	}
	return c.index.Search(c.name, query, topK)
}
