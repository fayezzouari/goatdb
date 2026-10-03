//go:build amd64

package core

import "golang.org/x/sys/cpu"

// The AVX2 kernels use VFMADD231PS, so FMA3 must be present as well.
var hasSIMD = cpu.X86.HasAVX2 && cpu.X86.HasFMA

//go:noescape
func dotSIMD(a, b *float32, n int) float32

//go:noescape
func l2SqSIMD(a, b *float32, n int) float32
