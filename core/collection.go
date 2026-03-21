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
