package index

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
	"slices"
	"strconv"
	"testing"

	"github.com/fayezzouari/goatdb/core"
)

type testIndex interface {
	core.Index
	core.Persistable
}

var allMetrics = []core.DistanceMetric{core.Cosine, core.Euclidean, core.DotProduct, core.Manhattan}

func newTestIndex(kind string, dim int, metric core.DistanceMetric, vecs []core.Vector) testIndex {
	switch kind {
	case "flat":
		return NewFlatIndex(dim, metric)
	case "hnsw":
		return NewHNSWIndex(dim, 8, 64, 64, metric)
	case "lsh":
		return NewLSHIndex(dim, 4, 4, metric)
	case "ivf":
		idx := NewIVFIndex(dim, 4, 4, metric)
		if len(vecs) > 0 {
			idx.Train(vecs)
		}
		return idx
	}
	panic(kind)
}

func randomVectors(r *rand.Rand, n, dim int) []core.Vector {
	vecs := make([]core.Vector, n)
	for i := range vecs {
		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = (r.Float32()*2 - 1) * 3
		}
		vecs[i] = core.Vector{Embeddings: emb}
	}
	return vecs
}

func closeTo(got, want float32) bool {
	return math.Abs(float64(got-want)) <= 1e-4*math.Max(1, math.Abs(float64(want)))
}

func TestSearchDistancesMatchMetric(t *testing.T) {
	const n, dim = 200, 21
	for _, kind := range []string{"flat", "hnsw", "lsh", "ivf"} {
		for _, metric := range allMetrics {
			t.Run(kind+"/"+string(metric), func(t *testing.T) {
				r := rand.New(rand.NewSource(1))
				vecs := randomVectors(r, n, dim)
				orig := make([][]float32, n)
				for i, v := range vecs {
					orig[i] = slices.Clone(v.Embeddings)
				}
				idx := newTestIndex(kind, dim, metric, vecs)
				for i, v := range vecs {
					idx.AddVector(strconv.Itoa(i), v)
				}
				query := randomVectors(r, 1, dim)[0]
				qOrig := slices.Clone(query.Embeddings)

				results := idx.Search(query, 10)
				if len(results) == 0 {
					t.Fatal("no results")
				}
				for _, res := range results {
					i, _ := strconv.Atoi(res.Id)
					want := core.DistSlices(qOrig, orig[i], metric)
					if !closeTo(res.Distance, want) {
						t.Errorf("id %s: distance %v, want %v", res.Id, res.Distance, want)
					}
				}
				if !slices.Equal(query.Embeddings, qOrig) {
					t.Error("query slice was modified")
				}
				for i, v := range vecs {
					if !slices.Equal(v.Embeddings, orig[i]) {
						t.Fatalf("input vector %d was modified", i)
					}
				}
			})
		}
	}
}

func TestSearchOrderMatchesBruteForce(t *testing.T) {
	const n, dim, topK = 300, 16, 5
	for _, metric := range allMetrics {
		t.Run(string(metric), func(t *testing.T) {
			r := rand.New(rand.NewSource(2))
			vecs := randomVectors(r, n, dim)
			idx := NewFlatIndex(dim, metric)
			for i, v := range vecs {
				idx.AddVector(strconv.Itoa(i), v)
			}
			q := randomVectors(r, 1, dim)[0]

			type pair struct {
				id   string
				dist float32
			}
			brute := make([]pair, n)
			for i, v := range vecs {
				brute[i] = pair{strconv.Itoa(i), core.DistSlices(q.Embeddings, v.Embeddings, metric)}
			}
			slices.SortFunc(brute, func(a, b pair) int {
				if a.dist < b.dist {
					return -1
				}
				if a.dist > b.dist {
					return 1
				}
				return 0
			})

			got := idx.Search(q, topK)
			slices.SortFunc(got, func(a, b core.SearchResult) int {
				if a.Distance < b.Distance {
					return -1
				}
				if a.Distance > b.Distance {
					return 1
				}
				return 0
			})
			for i := range got {
				if !closeTo(got[i].Distance, brute[i].dist) {
					t.Errorf("rank %d: distance %v, want %v", i, got[i].Distance, brute[i].dist)
				}
			}
		})
	}
}

func TestSaveLoadPreservesDistances(t *testing.T) {
	const n, dim = 150, 12
	for _, kind := range []string{"flat", "hnsw", "lsh", "ivf"} {
		for _, metric := range []core.DistanceMetric{core.Cosine, core.Euclidean} {
			t.Run(kind+"/"+string(metric), func(t *testing.T) {
				r := rand.New(rand.NewSource(3))
				vecs := randomVectors(r, n, dim)
				idx := newTestIndex(kind, dim, metric, vecs)
				for i, v := range vecs {
					idx.AddVector(strconv.Itoa(i), v)
				}
				path := t.TempDir() + "/idx.bin"
				if err := idx.Save(path); err != nil {
					t.Fatal(err)
				}
				idx2 := newTestIndex(kind, dim, metric, nil)
				if err := idx2.Load(path); err != nil {
					t.Fatal(err)
				}
				q := randomVectors(r, 1, dim)[0]
				byID := func(rs []core.SearchResult) map[string]float32 {
					m := make(map[string]float32, len(rs))
					for _, res := range rs {
						m[res.Id] = res.Distance
					}
					return m
				}
				before := byID(idx.Search(q, 10))
				after := byID(idx2.Search(q, 10))
				if len(before) != len(after) {
					t.Fatalf("got %d results after load, want %d", len(after), len(before))
				}
				for id, d := range before {
					if after[id] != d {
						t.Errorf("id %s: distance %v after load, want %v", id, after[id], d)
					}
				}
			})
		}
	}
}

func TestFlatLoadLegacyCosineFile(t *testing.T) {
	path := t.TempDir() + "/flat.bin"
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	raw := map[string][]float32{"a": {3, 4}, "b": {0, 2}}
	if err := gob.NewEncoder(f).Encode(flatState{Dim: 2, DistanceMetric: core.Cosine, Vectors: raw}); err != nil {
		t.Fatal(err)
	}
	f.Close()

	idx := NewFlatIndex(2, core.Cosine)
	if err := idx.Load(path); err != nil {
		t.Fatal(err)
	}
	q := []float32{1, 0}
	for _, res := range idx.Search(core.Vector{Embeddings: q}, 2) {
		want := core.DistSlices(q, raw[res.Id], core.Cosine)
		if !closeTo(res.Distance, want) {
			t.Errorf("id %s: distance %v, want %v", res.Id, res.Distance, want)
		}
	}
}
