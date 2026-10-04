package index

import "testing"

func TestSegArrayLocate(t *testing.T) {
	for _, stride := range []int{1, 33, 128, 1536, 1 << 14} {
		a := newSegArray[float32](stride)
		const n = 300_000
		// Keep the allocation small for wide strides.
		if stride >= 1<<14 {
			a.grow(100)
		} else if stride >= 1536 {
			a.grow(2000)
		} else {
			a.grow(n)
		}
		// Every slot must map to a distinct, in-bounds position, in order.
		prevSeg, prevOff := 0, -1
		for slot := 0; slot < a.len(); slot++ {
			seg, off := a.locate(slot)
			if seg >= len(a.segs) || (off+1)*stride > len(a.segs[seg]) {
				t.Fatalf("stride %d slot %d: out of bounds (seg %d off %d)", stride, slot, seg, off)
			}
			switch {
			case seg == prevSeg && off == prevOff+1:
			case seg == prevSeg+1 && off == 0 && (prevOff+1)*stride == len(a.segs[prevSeg]):
			default:
				t.Fatalf("stride %d slot %d: got seg %d off %d after seg %d off %d", stride, slot, seg, off, prevSeg, prevOff)
			}
			prevSeg, prevOff = seg, off
		}
	}
}

func TestSegArrayStableOnGrow(t *testing.T) {
	a := newSegArray[int32](4)
	a.grow(10)
	first := a.at(3)
	first[0] = 42
	a.grow(1 << 18)
	if a.at(3)[0] != 42 || &a.at(3)[0] != &first[0] {
		t.Fatal("slot moved when the array grew")
	}
}
