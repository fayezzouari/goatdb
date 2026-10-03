package storage

import "testing"

// ── Batched hydration (100 ids, dim=768) ──────────────────────────────────────

func BenchmarkStoreHydrate100(b *testing.B) {
	const dim, n, k = 768, 1000, 100
	s, err := Open(b.TempDir(), dim)
	if err != nil {
		b.Fatal(err)
	}
	b.Cleanup(func() { s.Close() })
	for i := 0; i < n; i++ {
		s.Add(benchID(i), randomEmbeddings(dim), map[string]any{"title": benchID(i), "rank": i})
	}
	ids := make([]string, k)
	for i := range ids {
		ids[i] = benchID(i * (n / k))
	}

	b.Run("get_loop", func(b *testing.B) {
		b.ReportAllocs()
		for i := 0; i < b.N; i++ {
			for _, id := range ids {
				if _, _, err := s.Get(id); err != nil {
					b.Fatal(err)
				}
			}
		}
	})
	for _, tc := range []struct {
		name            string
		withEmb, withMd bool
	}{
		{"getmany_vectors_metadata", true, true},
		{"getmany_metadata", false, true},
		{"getmany_ids_only", false, false},
	} {
		tc := tc
		b.Run(tc.name, func(b *testing.B) {
			b.ReportAllocs()
			for i := 0; i < b.N; i++ {
				if _, err := s.GetMany(ids, tc.withEmb, tc.withMd); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
