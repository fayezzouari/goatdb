//go:build arm64

package core

// ASIMD (NEON) is mandatory on arm64.
const hasSIMD = true

//go:noescape
func dotSIMD(a, b *float32, n int) float32

//go:noescape
func l2SqSIMD(a, b *float32, n int) float32
