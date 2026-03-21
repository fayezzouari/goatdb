package core

type Index interface {
	AddVector(id string, vector Vector)
	GetVector(id string) (Vector, bool)
	DeleteVector(id string) bool
	Search(query Vector, topK int) []SearchResult
}

type Persistable interface {
	Save(path string) error
	Load(path string) error
}
