//go:build !amd64 && !arm64

package core

const hasSIMD = false

func dotSIMD(a, b *float32, n int) float32  { panic("unreachable") }
func l2SqSIMD(a, b *float32, n int) float32 { panic("unreachable") }
