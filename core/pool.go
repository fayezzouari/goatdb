package core

type VectorPool struct {
	data  []float32
	live  []bool
	dim   int
	slots int
	free  []int32
}

func NewVectorPool(dim, initialCap int) *VectorPool {
	return &VectorPool{
		dim:  dim,
		data: make([]float32, 0, initialCap*dim),
		live: make([]bool, 0, initialCap),
	}
}

func (p *VectorPool) Add(emb []float32) int32 {
	if len(p.free) > 0 {
		idx := p.free[len(p.free)-1]
		p.free = p.free[:len(p.free)-1]
		copy(p.data[int(idx)*p.dim:], emb[:p.dim])
		p.live[idx] = true
		return idx
	}
	idx := int32(p.slots)
	p.slots++
	p.data = append(p.data, emb[:p.dim]...)
	p.live = append(p.live, true)
	return idx
}

func (p *VectorPool) Get(idx int32) []float32 {
	start := int(idx) * p.dim
	return p.data[start : start+p.dim]
}

func (p *VectorPool) Free(idx int32) {
	p.live[idx] = false
	p.free = append(p.free, idx)
}

func (p *VectorPool) ForEach(fn func(idx int32, emb []float32)) {
	for i := 0; i < p.slots; i++ {
		if p.live[i] {
			start := i * p.dim
			fn(int32(i), p.data[start:start+p.dim])
		}
	}
}

func (p *VectorPool) Len() int {
	return p.slots - len(p.free)
}

func (p *VectorPool) Slots() int {
	return p.slots
}
