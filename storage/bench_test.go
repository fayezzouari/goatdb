package storage

import (
	"fmt"
	"math/rand"
	"testing"
)

func randomEmbeddings(dim int) []float32 {
	emb := make([]float32, dim)
	for i := range emb {
		emb[i] = rand.Float32()*2 - 1
	}
	return emb
}

func benchID(i int) string {
	return fmt.Sprintf("vec-%08d", i)
}

// ── VectorStore ───────────────────────────────────────────────────────────────

func BenchmarkVectorStoreWrite(b *testing.B) {
	for _, dim := range []int{64, 128, 512, 1536} {
		dim := dim
		path := b.TempDir() + "/vectors.bin"
		vs, err := openVectorStore(path, dim)
		if err != nil {
			b.Fatal(err)
		}
		emb := randomEmbeddings(dim)
		b.Run(fmt.Sprintf("dim%d", dim), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				vs.Write(i%1024, emb)
			}
		})
		vs.Close()
	}
}

func BenchmarkVectorStoreRead(b *testing.B) {
	for _, dim := range []int{64, 128, 512, 1536} {
		dim := dim
		path := b.TempDir() + "/vectors.bin"
		vs, err := openVectorStore(path, dim)
		if err != nil {
			b.Fatal(err)
		}
		emb := randomEmbeddings(dim)
		for i := 0; i < 1024; i++ {
			vs.Write(i, emb)
		}
		b.Run(fmt.Sprintf("dim%d", dim), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				vs.Read(i % 1024)
			}
		})
		vs.Close()
	}
}

// ── MetaStore ─────────────────────────────────────────────────────────────────

func BenchmarkMetaStoreAllocSlot(b *testing.B) {
	ms, err := openMetaStore(b.TempDir() + "/meta.db")
	if err != nil {
		b.Fatal(err)
	}
	defer ms.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ms.AllocSlot(benchID(i))
	}
}

func BenchmarkMetaStoreGetSlot(b *testing.B) {
	ms, err := openMetaStore(b.TempDir() + "/meta.db")
	if err != nil {
		b.Fatal(err)
	}
	defer ms.Close()
	for i := 0; i < 1000; i++ {
		ms.AllocSlot(benchID(i))
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ms.GetSlot(benchID(i % 1000))
	}
}

func BenchmarkMetaStorePutMeta(b *testing.B) {
	ms, err := openMetaStore(b.TempDir() + "/meta.db")
	if err != nil {
		b.Fatal(err)
	}
	defer ms.Close()
	meta := map[string]any{"label": "bench", "score": 1.0, "active": true}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ms.PutMeta(benchID(i%1000), meta)
	}
}

// ── Store (full stack) ────────────────────────────────────────────────────────

func BenchmarkStoreAdd(b *testing.B) {
	for _, dim := range []int{128, 1536} {
		dim := dim
		s, err := Open(b.TempDir(), dim)
		if err != nil {
			b.Fatal(err)
		}
		emb := randomEmbeddings(dim)
		meta := map[string]any{"label": "bench"}
		next := 0
		b.Run(fmt.Sprintf("dim%d", dim), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if err := s.Add(benchID(next), emb, meta); err != nil {
					b.Fatal(err)
				}
				next++
			}
		})
		s.Close()
	}
}

func BenchmarkStoreAddBatch(b *testing.B) {
	const batchSize = 1000
	for _, dim := range []int{128, 1536} {
		dim := dim
		s, err := Open(b.TempDir(), dim)
		if err != nil {
			b.Fatal(err)
		}
		emb := randomEmbeddings(dim)
		meta := map[string]any{"label": "bench"}
		next := 0
		b.Run(fmt.Sprintf("dim%d/batch%d", dim, batchSize), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				batch := make([]StoredVector, batchSize)
				for j := range batch {
					batch[j] = StoredVector{Id: benchID(next), Embeddings: emb, Metadata: meta}
					next++
				}
				if err := s.AddBatch(batch); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(b.N*batchSize)/b.Elapsed().Seconds(), "vectors/s")
		})
		s.Close()
	}
}

func BenchmarkWALReplay(b *testing.B) {
	const records = 10000
	for _, dim := range []int{128, 1536} {
		dim := dim
		path := b.TempDir() + "/wal.log"
		wal, _, err := openWAL(path)
		if err != nil {
			b.Fatal(err)
		}
		emb := randomEmbeddings(dim)
		for i := 0; i < records; i++ {
			wal.Append(opInsert, benchID(i), uint32(i), emb)
		}
		wal.Close()
		b.Run(fmt.Sprintf("dim%d/n%d", dim, records), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				w, entries, err := openWAL(path)
				if err != nil || len(entries) != records {
					b.Fatalf("replayed %d entries: %v", len(entries), err)
				}
				w.Close()
			}
		})
	}
}

func BenchmarkStoreGet(b *testing.B) {
	for _, dim := range []int{128, 1536} {
		dim := dim
		s, err := Open(b.TempDir(), dim)
		if err != nil {
			b.Fatal(err)
		}
		emb := randomEmbeddings(dim)
		for i := 0; i < 1000; i++ {
			s.Add(benchID(i), emb, nil)
		}
		b.Run(fmt.Sprintf("dim%d", dim), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				s.Get(benchID(i % 1000))
			}
		})
		s.Close()
	}
}

func BenchmarkStoreDelete(b *testing.B) {
	s, err := Open(b.TempDir(), 128)
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()
	emb := randomEmbeddings(128)
	for i := 0; i < b.N; i++ {
		s.Add(benchID(i), emb, nil)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Delete(benchID(i))
	}
}

func BenchmarkStoreParallelGet(b *testing.B) {
	s, err := Open(b.TempDir(), 128)
	if err != nil {
		b.Fatal(err)
	}
	defer s.Close()
	emb := randomEmbeddings(128)
	for i := 0; i < 1000; i++ {
		s.Add(benchID(i), emb, nil)
	}
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			s.Get(benchID(i % 1000))
			i++
		}
	})
}
