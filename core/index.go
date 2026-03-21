package core

type Index interface {
	AddVector(collectionName string, id string, vector Vector)
	GetVector(collectionName string, id string) (Vector, bool)
	DeleteVector(collectionName string, id string) bool
	Search(collectionName string, query Vector, topK int) []SearchResult
	Save(path string) error
	Load(path string) error
}
