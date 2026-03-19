package core

type Collection struct {
	id             int
	name           string
	index          string
	vectors        map[string]Vector
	distanceMetric string
	dim            int
}

func (c *Collection) AddVector(id string, vector Vector) {
	c.vectors[id] = vector
}

func (c *Collection) GetVector(id string) (Vector, bool) {
	vector, exists := c.vectors[id]
	return vector, exists
}

func (c *Collection) DeleteVector(id string) {
	delete(c.vectors, id)
}
