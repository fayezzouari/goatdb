package core

type Collection struct {
	id             int
	name           string
	index          string
	vectors        map[string]Vector
	distanceMetric string
	dim            int
}
