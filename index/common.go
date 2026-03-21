package index

import "github.com/fayez/goatdb/core"

type resultHeap []core.SearchResult

func (h resultHeap) Len() int            { return len(h) }
func (h resultHeap) Less(i, j int) bool  { return h[i].Distance > h[j].Distance }
func (h resultHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *resultHeap) Push(x any)         { *h = append(*h, x.(core.SearchResult)) }
func (h *resultHeap) Pop() any           { old := *h; n := len(old); x := old[n-1]; *h = old[:n-1]; return x }
