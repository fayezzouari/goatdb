package core

import (
	"math"
	"math/rand"
	"testing"
)

func dotRef(a, b []float32) (sum, mag float64) {
	for i := range a {
		p := float64(a[i]) * float64(b[i])
		sum += p
		mag += math.Abs(p)
	}
	return sum, mag
}

func l2SqRef(a, b []float32) float64 {
	var s float64
	for i := range a {
		d := float64(a[i]) - float64(b[i])
		s += d * d
	}
	return s
}

// within allows float32 rounding error proportional to the magnitude of the
// summed terms, since kernels may reorder and fuse the additions.
func within(got float32, want, mag float64) bool {
	return math.Abs(float64(got)-want) <= 1e-5*mag+1e-6
}

func randSlice(r *rand.Rand, n int) []float32 {
	s := make([]float32, n)
	for i := range s {
		s[i] = r.Float32()*4 - 2
	}
	return s
}

type kernel struct {
	name string
	dot  func(a, b []float32) float32
	l2Sq func(a, b []float32) float32
}

func kernels() []kernel {
	ks := []kernel{
		{"scalar", dotScalar, l2SqScalar},
		{"dispatch", Dot, L2SqSlices},
	}
	if hasSIMD {
		ks = append(ks, kernel{
			"simd",
			func(a, b []float32) float32 {
				if len(a) == 0 {
					return 0
				}
				return dotSIMD(&a[0], &b[0], len(a))
			},
			func(a, b []float32) float32 {
				if len(a) == 0 {
					return 0
				}
				return l2SqSIMD(&a[0], &b[0], len(a))
			},
		})
	}
	return ks
}

func checkKernels(t *testing.T, a, b []float32) {
	t.Helper()
	wantDot, magDot := dotRef(a, b)
	wantL2 := l2SqRef(a, b)
	for _, k := range kernels() {
		if got := k.dot(a, b); !within(got, wantDot, magDot) {
			t.Errorf("%s dot n=%d: got %v, want %v", k.name, len(a), got, wantDot)
		}
		if got := k.l2Sq(a, b); !within(got, wantL2, wantL2) {
			t.Errorf("%s l2Sq n=%d: got %v, want %v", k.name, len(a), got, wantL2)
		}
	}
}

func TestKernelsMatchReference(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	for n := 0; n <= 300; n++ {
		checkKernels(t, randSlice(r, n), randSlice(r, n))
	}
}

func TestKernelsUnaligned(t *testing.T) {
	r := rand.New(rand.NewSource(2))
	a, b := randSlice(r, 140), randSlice(r, 140)
	for off := 1; off < 4; off++ {
		for _, n := range []int{1, 7, 33, 100, 131} {
			checkKernels(t, a[off:off+n], b[4-off:4-off+n])
		}
	}
}

func TestKernelsExactSmallIntegers(t *testing.T) {
	for n := 0; n <= 70; n++ {
		a := make([]float32, n)
		b := make([]float32, n)
		var wantDot, wantL2 float32
		for i := range a {
			a[i] = float32(i%7 - 3)
			b[i] = float32(i%5 - 2)
			wantDot += a[i] * b[i]
			d := a[i] - b[i]
			wantL2 += d * d
		}
		for _, k := range kernels() {
			if got := k.dot(a, b); got != wantDot {
				t.Errorf("%s dot n=%d: got %v, want %v", k.name, n, got, wantDot)
			}
			if got := k.l2Sq(a, b); got != wantL2 {
				t.Errorf("%s l2Sq n=%d: got %v, want %v", k.name, n, got, wantL2)
			}
		}
	}
}

func FuzzKernels(f *testing.F) {
	for _, n := range []int{0, 1, 3, 4, 15, 16, 31, 32, 33, 64, 129, 300} {
		f.Add(int64(n), uint16(n))
	}
	f.Fuzz(func(t *testing.T, seed int64, n uint16) {
		r := rand.New(rand.NewSource(seed))
		size := int(n) % 2048
		checkKernels(t, randSlice(r, size), randSlice(r, size))
	})
}
