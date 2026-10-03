package index

import (
	"math/rand"

	"github.com/fayezzouari/goatdb/core"
)

// prepareVectors returns vectors in the form m.Dist expects. When the metric
// needs normalized copies, at most limit vectors are sampled (limit <= 0
// keeps all) so training sets are not duplicated in full.
func prepareVectors(m core.Metric, vectors []core.Vector, limit int) []core.Vector {
	if !m.Normalizes() {
		return vectors
	}
	src := vectors
	if limit > 0 && len(src) > limit {
		src = make([]core.Vector, limit)
		for i, p := range rand.Perm(len(vectors))[:limit] {
			src[i] = vectors[p]
		}
	}
	out := make([]core.Vector, len(src))
	for i, v := range src {
		out[i] = core.Vector{Embeddings: m.Prepare(v.Embeddings)}
	}
	return out
}
