package index

import (
	"math/bits"
	"unsafe"
)

// Segment sizing for segArray. The first segment holds segBaseSlots slots and
// the following ones double in size until a segment reaches about
// segTargetBytes; from then on every segment has that size. Small indexes
// therefore stay small, and large ones grow without ever copying their data.
const (
	segBaseBits    = 6
	segTargetBytes = 4 << 20
	segMaxBits     = 16
)

// segArray is a slot-indexed array with stride elements per slot, stored in
// segments that never move once allocated. Growing allocates a new segment
// and copies only the directory of segment headers, so it costs no
// transient second copy of the data (as append on a flat slice does), and
// slices returned by at stay valid while the array grows.
//
// segArray does no locking; callers serialize grow against other calls.
type segArray[T any] struct {
	stride   int
	baseBits uint
	maxBits  uint
	// lastDoubling is the index of the last doubling segment, which covers
	// slots [1<<maxBits, 2<<maxBits).
	lastDoubling int
	segs         [][]T
	slots        int
}

func newSegArray[T any](stride int) segArray[T] {
	var zero T
	slotBytes := stride * int(unsafe.Sizeof(zero))
	maxBits := uint(segMaxBits)
	for maxBits > 0 && slotBytes<<maxBits > segTargetBytes {
		maxBits--
	}
	baseBits := uint(segBaseBits)
	if baseBits > maxBits {
		baseBits = maxBits
	}
	return segArray[T]{
		stride:       stride,
		baseBits:     baseBits,
		maxBits:      maxBits,
		lastDoubling: int(maxBits-baseBits) + 1,
	}
}

// locate maps a slot to its segment and the slot's offset in that segment.
func (a *segArray[T]) locate(slot int) (seg, off int) {
	if slot>>(a.maxBits+1) == 0 {
		if slot>>a.baseBits == 0 {
			return 0, slot
		}
		seg = bits.Len(uint(slot) >> a.baseBits)
		return seg, slot - (1 << a.baseBits << (seg - 1))
	}
	r := slot - 2<<a.maxBits
	return a.lastDoubling + 1 + r>>a.maxBits, r & (1<<a.maxBits - 1)
}

func (a *segArray[T]) segSlots(seg int) int {
	switch {
	case seg == 0:
		return 1 << a.baseBits
	case seg <= a.lastDoubling:
		return 1 << a.baseBits << (seg - 1)
	default:
		return 1 << a.maxBits
	}
}

// grow makes slots [0, n) addressable. New slots are zero.
func (a *segArray[T]) grow(n int) {
	for a.slots < n {
		k := len(a.segs)
		size := a.segSlots(k)
		a.segs = append(a.segs, make([]T, size*a.stride))
		a.slots += size
	}
}

// at returns the stride elements of slot. The slot must be below len.
func (a *segArray[T]) at(slot int32) []T {
	seg, off := a.locate(int(slot))
	s := a.segs[seg]
	i := off * a.stride
	return s[i : i+a.stride : i+a.stride]
}

// one returns a pointer to the single element of slot in a stride-1 array.
func (a *segArray[T]) one(slot int32) *T {
	seg, off := a.locate(int(slot))
	return &a.segs[seg][off]
}

// len returns the number of addressable slots.
func (a *segArray[T]) len() int { return a.slots }
