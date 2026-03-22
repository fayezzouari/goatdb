//go:build amd64

package core

import "golang.org/x/sys/cpu"

var hasAVX2 = cpu.X86.HasAVX2

//go:noescape
func dotProductAVX2(a, b *float32, n int) float32

//go:noescape
func l2SquaredAVX2(a, b *float32, n int) float32
