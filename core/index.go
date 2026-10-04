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

type Trainable interface {
	Train(vectors []Vector)
}

// EfSearcher is implemented by indexes whose search depth can be set per
// query. For HNSW, ef is the beam width at layer 0 (the effective value is
// max(ef, topK)). For IVF, ef is the number of lists to probe. Implementations
// must not mutate shared state, so concurrent searches may use different ef.
type EfSearcher interface {
	SearchEf(query Vector, topK, ef int) []SearchResult
}

// ConcurrentAdder is implemented by indexes whose AddVector may be called
// from several goroutines at once and scales with them. Callers may insert
// a batch into such an index in parallel.
type ConcurrentAdder interface {
	ConcurrentAdd() bool
}
