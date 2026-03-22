//go:build !amd64

package core

const hasAVX2 = false

func dotProductAVX2(a, b *float32, n int) float32 { panic("unreachable") }
func l2SquaredAVX2(a, b *float32, n int) float32  { panic("unreachable") }
