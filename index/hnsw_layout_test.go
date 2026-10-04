package index

import (
	"encoding/gob"
	"fmt"
	"math/rand"
	"os"
	"testing"

	"github.com/fayezzouari/goatdb/core"
)

func randVecs(r *rand.Rand, n, dim int) []core.Vector {
	out := make([]core.Vector, n)
	for i := range out {
		emb := make([]float32, dim)
		for j := range emb {
			emb[j] = r.Float32()*2 - 1
		}
		out[i] = core.Vector{Embeddings: emb}
	}
	return out
}

// checkGraph verifies the structural invariants of the slot layout.
func checkGraph(t *testing.T, h *HNSWIndex) {
	t.Helper()
	h.mu.RLock()
	defer h.mu.RUnlock()
	live := 0
	for slot := int32(0); slot < h.nSlots.Load(); slot++ {
		lvl := h.level(slot)
		if lvl < 0 {
			continue
		}
		live++
		id := *h.slotIDs.one(slot)
		if got, ok := h.ids[id]; !ok || got != slot {
			t.Fatalf("slot %d: id %q maps to %d (ok=%v)", slot, id, got, ok)
		}
		for layer := 0; layer <= lvl; layer++ {
			blk := h.links(slot, layer)
			mMax := h.M
			if layer == 0 {
				mMax = 2 * h.M
			}
			if int(blk[0]) > mMax {
				t.Fatalf("slot %d layer %d: %d links > %d", slot, layer, blk[0], mMax)
			}
			for _, nb := range blk[1 : 1+blk[0]] {
				if nb == slot {
					t.Fatalf("slot %d layer %d: self link", slot, layer)
				}
				if nb >= h.nSlots.Load() {
					t.Fatalf("slot %d layer %d: link %d out of range", slot, layer, nb)
				}
			}
		}
	}
	if live != len(h.ids) {
		t.Fatalf("%d live slots, %d ids", live, len(h.ids))
	}
	if entry, maxLayer := h.entryPoint(); live > 0 && h.level(entry) != maxLayer {
		t.Fatalf("entry slot %d has level %d, maxLayer %d", entry, h.level(entry), maxLayer)
	}
}

func TestHNSWDeleteReuseSlots(t *testing.T) {
	const dim = 8
	r := rand.New(rand.NewSource(21))
	h := NewHNSWIndex(dim, 6, 40, 40, core.Euclidean)
	vecs := randVecs(r, 600, dim)
	for i, v := range vecs {
		h.AddVector(fmt.Sprintf("v%d", i), v)
	}
	// Delete every other vector, then insert as many new ones: they reuse
	// the freed slots while stale links to those slots remain.
	for i := 0; i < len(vecs); i += 2 {
		if !h.DeleteVector(fmt.Sprintf("v%d", i)) {
			t.Fatalf("delete v%d failed", i)
		}
	}
	slotsBefore := h.nSlots.Load()
	fresh := randVecs(r, 300, dim)
	for i, v := range fresh {
		h.AddVector(fmt.Sprintf("n%d", i), v)
	}
	if h.nSlots.Load() != slotsBefore {
		t.Errorf("slots grew from %d to %d; freed slots were not reused", slotsBefore, h.nSlots.Load())
	}
	checkGraph(t, h)
	for i := 0; i < len(vecs); i += 2 {
		if _, ok := h.GetVector(fmt.Sprintf("v%d", i)); ok {
			t.Fatalf("deleted v%d still present", i)
		}
	}
	for i, v := range fresh {
		res := h.SearchEf(v, 1, 64)
		if len(res) != 1 || res[0].Id != fmt.Sprintf("n%d", i) {
			t.Fatalf("search n%d: got %v", i, res)
		}
	}
	// Delete everything, including the entry point, and start over.
	for i := 1; i < len(vecs); i += 2 {
		h.DeleteVector(fmt.Sprintf("v%d", i))
	}
	for i := range fresh {
		h.DeleteVector(fmt.Sprintf("n%d", i))
	}
	if res := h.Search(vecs[0], 1); res != nil {
		t.Fatalf("empty index returned %v", res)
	}
	h.AddVector("again", vecs[0])
	if res := h.Search(vecs[0], 1); len(res) != 1 || res[0].Id != "again" {
		t.Fatalf("got %v", res)
	}
}

func TestHNSWSaveLoadRoundTrip(t *testing.T) {
	for _, metric := range []core.DistanceMetric{core.Euclidean, core.Cosine} {
		t.Run(string(metric), func(t *testing.T) {
			const dim = 12
			r := rand.New(rand.NewSource(5))
			h := NewHNSWIndex(dim, 8, 60, 50, metric)
			vecs := randVecs(r, 1500, dim)
			for i, v := range vecs {
				h.AddVector(fmt.Sprintf("v%d", i), v)
			}
			for i := 0; i < 1500; i += 7 {
				h.DeleteVector(fmt.Sprintf("v%d", i))
			}
			h.Train(vecs)
			path := t.TempDir() + "/hnsw.bin"
			if err := h.Save(path); err != nil {
				t.Fatal(err)
			}
			h2 := NewHNSWIndex(dim, 4, 10, 10, core.DotProduct)
			if err := h2.Load(path); err != nil {
				t.Fatal(err)
			}
			checkGraph(t, h2)
			if h2.nSlots.Load() != h.nSlots.Load() || len(h2.free) != len(h.free) || h2.M != 8 || h2.ef != 50 || h2.distanceMetric != metric {
				t.Fatalf("loaded index differs: slots %d/%d free %d/%d", h2.nSlots.Load(), h.nSlots.Load(), len(h2.free), len(h.free))
			}
			if h2.codebook == nil || *h2.codebook != *h.codebook {
				t.Fatal("codebook not restored")
			}
			for slot := int32(0); slot < h.nSlots.Load(); slot++ {
				if h.level(slot) != h2.level(slot) {
					t.Fatalf("slot %d: level %d != %d", slot, h.level(slot), h2.level(slot))
				}
				for layer := 0; layer <= h.level(slot); layer++ {
					a, b := h.links(slot, layer), h2.links(slot, layer)
					for i := 0; i <= int(a[0]); i++ {
						if a[i] != b[i] {
							t.Fatalf("slot %d layer %d differs", slot, layer)
						}
					}
				}
			}
			for i, v := range vecs[:200] {
				a, b := h.Search(v, 5), h2.Search(v, 5)
				if len(a) != len(b) {
					t.Fatalf("query %d: %d vs %d results", i, len(a), len(b))
				}
				for k := range a {
					if a[k].Id != b[k].Id || a[k].Distance != b[k].Distance {
						t.Fatalf("query %d result %d: %v vs %v", i, k, a[k], b[k])
					}
				}
			}
		})
	}
}

// TestHNSWLoadLegacyGob checks that an index saved in the version 1 gob
// format loads into the slot layout with the same graph.
func TestHNSWLoadLegacyGob(t *testing.T) {
	const dim = 6
	r := rand.New(rand.NewSource(8))
	h := NewHNSWIndex(dim, 6, 40, 40, core.Cosine)
	vecs := randVecs(r, 400, dim)
	for i, v := range vecs {
		h.AddVector(fmt.Sprintf("v%d", i), v)
	}
	// Write the graph the way version 1 did.
	nodes := make(map[string]hnswNodeState, len(h.ids))
	for id, slot := range h.ids {
		conns := make([][]string, h.level(slot)+1)
		for l := range conns {
			blk := h.links(slot, l)
			for _, nb := range blk[1 : 1+blk[0]] {
				conns[l] = append(conns[l], *h.slotIDs.one(nb))
			}
		}
		nodes[id] = hnswNodeState{Embeddings: append([]float32(nil), h.vecs.at(slot)...), Connections: conns}
	}
	entry, maxLayer := h.entryPoint()
	state := hnswState{
		Dim: dim, M: 6, EfConstruction: 40, Ef: 40, ML: h.mL,
		DistanceMetric: core.Cosine,
		EntryPoint:     *h.slotIDs.one(entry),
		MaxLayer:       maxLayer,
		Nodes:          nodes,
		Normalized:     true,
	}
	path := t.TempDir() + "/hnsw.bin"
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	if err := gob.NewEncoder(f).Encode(state); err != nil {
		t.Fatal(err)
	}
	f.Close()

	h2 := NewHNSWIndex(dim, 6, 40, 40, core.Cosine)
	if err := h2.Load(path); err != nil {
		t.Fatal(err)
	}
	checkGraph(t, h2)
	if _, ml2 := h2.entryPoint(); ml2 != maxLayer || len(h2.ids) != len(h.ids) {
		t.Fatalf("maxLayer %d/%d ids %d/%d", ml2, maxLayer, len(h2.ids), len(h.ids))
	}
	for i, v := range vecs[:100] {
		a, b := h.Search(v, 5), h2.Search(v, 5)
		for k := range a {
			if a[k].Id != b[k].Id {
				t.Fatalf("query %d result %d: %v vs %v", i, k, a[k], b[k])
			}
		}
	}
	// A legacy index saves in the current format and loads back.
	path2 := t.TempDir() + "/hnsw2.bin"
	if err := h2.Save(path2); err != nil {
		t.Fatal(err)
	}
	h3 := NewHNSWIndex(dim, 6, 40, 40, core.Cosine)
	if err := h3.Load(path2); err != nil {
		t.Fatal(err)
	}
	checkGraph(t, h3)
}

func TestHNSWLoadCorrupt(t *testing.T) {
	const dim = 4
	r := rand.New(rand.NewSource(2))
	h := NewHNSWIndex(dim, 4, 20, 20, core.Euclidean)
	for i, v := range randVecs(r, 50, dim) {
		h.AddVector(fmt.Sprintf("v%d", i), v)
	}
	path := t.TempDir() + "/hnsw.bin"
	if err := h.Save(path); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, n := range []int{len(data) / 2, 20, 9} {
		if err := os.WriteFile(path, data[:n], 0o644); err != nil {
			t.Fatal(err)
		}
		if err := NewHNSWIndex(dim, 4, 20, 20, core.Euclidean).Load(path); err == nil {
			t.Errorf("truncated to %d bytes: expected an error", n)
		}
	}
}
