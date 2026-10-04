package index

import (
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/gob"
	"errors"
	"fmt"
	"io"
	"math"
	"os"

	"github.com/fayezzouari/goatdb/core"
)

// On-disk formats.
//
// Version 1 (no header) is a gob-encoded hnswState: every node with its
// vector and its links as id strings. It is still read, but no longer
// written.
//
// Version 2 starts with hnswMagic and a uint32 version, followed by the
// header and one record per slot, in slot order:
//
//	level   int8 (-1 for a free slot; nothing else follows)
//	id      uvarint length + bytes
//	vector  dim little-endian float32 (prepared for the metric)
//	links   for each layer 0..level: uvarint count + count uvarint slots
//
// All integers are little-endian. Slots and links are stored as is, so a
// loaded index has exactly the saved graph and free list.
var hnswMagic = [8]byte{'G', 'O', 'A', 'T', 'H', 'N', 'S', 'W'}

const hnswFormatVersion = 2

type hnswNodeState struct {
	Embeddings  []float32
	Connections [][]string
}

// hnswState is the version 1 format. The PQ fields are kept so that files
// written by older versions still decode; they are ignored on Load.
type hnswState struct {
	Dim            int
	M              int
	EfConstruction int
	Ef             int
	ML             float64
	DistanceMetric core.DistanceMetric
	EntryPoint     string
	MaxLayer       int
	Nodes          map[string]hnswNodeState
	SQMin          float32
	SQScale        float32
	HasSQ          bool
	PQNSubs        int
	PQNCentroids   int
	PQCentroids    []float32
	HasPQ          bool
	Normalized     bool
}

type hnswHeader struct {
	Dim            uint32
	M              uint32
	EfConstruction uint32
	Ef             uint32
	ML             float64
	HasSQ          uint8
	SQMin          float32
	SQScale        float32
	MaxLayer       int32
	EntrySlot      int32
	NSlots         uint32
}

func (h *HNSWIndex) Save(path string) error {
	// The write lock keeps inserts, which run under the read lock, from
	// changing the graph while it is written.
	h.mu.Lock()
	defer h.mu.Unlock()
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(file, 1<<20)
	if err := h.writeTo(w); err != nil {
		file.Close()
		return err
	}
	if err := w.Flush(); err != nil {
		file.Close()
		return err
	}
	return file.Close()
}

func (h *HNSWIndex) writeTo(w *bufio.Writer) error {
	w.Write(hnswMagic[:])
	binary.Write(w, binary.LittleEndian, uint32(hnswFormatVersion))
	metric := []byte(h.distanceMetric)
	putUvarint(w, uint64(len(metric)))
	w.Write(metric)
	entry, maxLayer := h.entryPoint()
	nSlots := int(h.nSlots.Load())
	hdr := hnswHeader{
		Dim:            uint32(h.dim),
		M:              uint32(h.M),
		EfConstruction: uint32(h.efConstruction),
		Ef:             uint32(h.ef),
		ML:             h.mL,
		MaxLayer:       int32(maxLayer),
		EntrySlot:      entry,
		NSlots:         uint32(nSlots),
	}
	if h.codebook != nil {
		hdr.HasSQ = 1
		hdr.SQMin = h.codebook.Min
		hdr.SQScale = h.codebook.Scale
	}
	if err := binary.Write(w, binary.LittleEndian, &hdr); err != nil {
		return err
	}
	var buf [4]byte
	for slot := int32(0); slot < int32(nSlots); slot++ {
		level := h.level(slot)
		w.WriteByte(byte(int8(level)))
		if level < 0 {
			continue
		}
		id := *h.slotIDs.one(slot)
		putUvarint(w, uint64(len(id)))
		w.WriteString(id)
		for _, x := range h.vecs.at(slot) {
			binary.LittleEndian.PutUint32(buf[:], math.Float32bits(x))
			w.Write(buf[:])
		}
		for layer := 0; layer <= level; layer++ {
			blk := h.links(slot, layer)
			putUvarint(w, uint64(blk[0]))
			for _, nb := range blk[1 : 1+blk[0]] {
				putUvarint(w, uint64(nb))
			}
		}
	}
	// bufio.Writer keeps the first error; Flush reports it.
	return nil
}

func putUvarint(w *bufio.Writer, x uint64) {
	var b [binary.MaxVarintLen64]byte
	n := binary.PutUvarint(b[:], x)
	w.Write(b[:n])
}

func (h *HNSWIndex) Load(path string) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()
	r := bufio.NewReaderSize(file, 1<<20)

	// Build the new graph without holding the lock, then swap it in.
	n := &HNSWIndex{keepPruned: true}
	magic, err := r.Peek(len(hnswMagic))
	if err == nil && bytes.Equal(magic, hnswMagic[:]) {
		r.Discard(len(hnswMagic))
		err = n.readFrom(r)
	} else {
		err = n.readLegacy(r)
	}
	if err != nil {
		return err
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	h.dim = n.dim
	h.M = n.M
	h.efConstruction = n.efConstruction
	h.ef = n.ef
	h.mL = n.mL
	h.distanceMetric = n.distanceMetric
	h.metric = n.metric
	h.ids = n.ids
	h.slotIDs = n.slotIDs
	h.vecs = n.vecs
	h.links0 = n.links0
	h.upper = n.upper
	h.levels = n.levels
	h.locks = n.locks
	h.nSlots.Store(n.nSlots.Load())
	// hasFree reads free without holding mu.
	h.allocMu.Lock()
	h.free = n.free
	h.allocMu.Unlock()
	h.entry.Store(n.entry.Load())
	h.codebook = n.codebook
	h.sq = n.sq
	// Abandon any Train in progress: its slots refer to the old graph.
	h.training = false
	h.trainDirty = nil
	return nil
}

var errHNSWCorrupt = errors.New("hnsw: corrupt index file")

func (h *HNSWIndex) readFrom(r *bufio.Reader) error {
	var version uint32
	if err := binary.Read(r, binary.LittleEndian, &version); err != nil {
		return err
	}
	if version != hnswFormatVersion {
		return fmt.Errorf("hnsw: unsupported index file version %d", version)
	}
	mlen, err := binary.ReadUvarint(r)
	if err != nil {
		return err
	}
	if mlen > 64 {
		return errHNSWCorrupt
	}
	metric := make([]byte, mlen)
	if _, err := io.ReadFull(r, metric); err != nil {
		return err
	}
	var hdr hnswHeader
	if err := binary.Read(r, binary.LittleEndian, &hdr); err != nil {
		return err
	}
	if hdr.Dim == 0 || hdr.M == 0 || hdr.NSlots > math.MaxInt32 ||
		hdr.MaxLayer < -1 || hdr.MaxLayer > hnswMaxLevel ||
		(hdr.MaxLayer >= 0 && hdr.EntrySlot >= int32(hdr.NSlots)) {
		return errHNSWCorrupt
	}

	h.distanceMetric = core.DistanceMetric(metric)
	h.metric = core.ResolveMetric(h.distanceMetric)
	h.efConstruction = int(hdr.EfConstruction)
	h.ef = int(hdr.Ef)
	h.mL = hdr.ML
	h.reset(int(hdr.Dim), int(hdr.M), int(hdr.NSlots))
	nSlots := int(hdr.NSlots)
	h.vecs.grow(nSlots)
	h.links0.grow(nSlots)
	h.upper.grow(nSlots)
	h.levels.grow(nSlots)
	h.slotIDs.grow(nSlots)
	h.locks.grow(nSlots)
	h.nSlots.Store(int32(nSlots))
	if hdr.MaxLayer >= 0 {
		h.setEntryPoint(hdr.EntrySlot, int(hdr.MaxLayer))
	}

	vbuf := make([]byte, 4*h.dim)
	var idBuf []byte
	for slot := int32(0); slot < int32(nSlots); slot++ {
		lb, err := r.ReadByte()
		if err != nil {
			return err
		}
		level := int(int8(lb))
		if level < 0 {
			*h.levels.one(slot) = -1
			h.free = append(h.free, slot)
			continue
		}
		if level > hnswMaxLevel {
			return errHNSWCorrupt
		}
		idLen, err := binary.ReadUvarint(r)
		if err != nil {
			return err
		}
		if idLen > 1<<20 {
			return errHNSWCorrupt
		}
		idBuf = slicesGrow(idBuf, int(idLen))
		if _, err := io.ReadFull(r, idBuf); err != nil {
			return err
		}
		id := string(idBuf)
		if _, err := io.ReadFull(r, vbuf); err != nil {
			return err
		}
		v := h.vecs.at(slot)
		for i := range v {
			v[i] = math.Float32frombits(binary.LittleEndian.Uint32(vbuf[i*4:]))
		}
		*h.slotIDs.one(slot) = id
		*h.levels.one(slot) = int8(level)
		if level > 0 {
			*h.upper.one(slot) = make([]int32, level*(h.M+1))
		}
		h.ids[id] = slot
		for layer := 0; layer <= level; layer++ {
			blk := h.links(slot, layer)
			cnt, err := binary.ReadUvarint(r)
			if err != nil {
				return err
			}
			if cnt > uint64(len(blk)-1) {
				return errHNSWCorrupt
			}
			for i := 1; i <= int(cnt); i++ {
				nb, err := binary.ReadUvarint(r)
				if err != nil {
					return err
				}
				if nb >= uint64(nSlots) {
					return errHNSWCorrupt
				}
				blk[i] = int32(nb)
			}
			blk[0] = int32(cnt)
		}
	}
	if entry, maxLayer := h.entryPoint(); maxLayer >= 0 && h.level(entry) != maxLayer {
		return errHNSWCorrupt
	}
	if hdr.HasSQ != 0 {
		h.setCodebook(&core.SQCodebook{Min: hdr.SQMin, Scale: hdr.SQScale, Dim: h.dim})
	}
	return nil
}

func slicesGrow(b []byte, n int) []byte {
	if cap(b) < n {
		return make([]byte, n)
	}
	return b[:n]
}

// setCodebook installs cb and quantizes every live vector with it.
func (h *HNSWIndex) setCodebook(cb *core.SQCodebook) {
	sq := newSegArray[int8](h.dim)
	nSlots := int(h.nSlots.Load())
	sq.grow(nSlots)
	for slot := int32(0); slot < int32(nSlots); slot++ {
		if h.level(slot) >= 0 {
			quantizeInto(cb, sq.at(slot), h.vecs.at(slot))
		}
	}
	h.codebook = cb
	h.sq = sq
}

// readLegacy loads the version 1 gob format.
func (h *HNSWIndex) readLegacy(r io.Reader) error {
	var s hnswState
	if err := gob.NewDecoder(r).Decode(&s); err != nil {
		return err
	}
	if s.Dim <= 0 || s.M <= 0 {
		return errHNSWCorrupt
	}
	h.efConstruction = s.EfConstruction
	h.ef = s.Ef
	h.mL = s.ML
	h.distanceMetric = s.DistanceMetric
	h.metric = core.ResolveMetric(s.DistanceMetric)
	h.reset(s.Dim, s.M, len(s.Nodes))

	for id, ns := range s.Nodes {
		if len(ns.Embeddings) != s.Dim || len(ns.Connections) == 0 || len(ns.Connections) > hnswMaxLevel+1 {
			return errHNSWCorrupt
		}
		slot := h.allocSlot(false)
		level := len(ns.Connections) - 1
		h.initNode(slot, id, ns.Embeddings, level)
		if s.Normalized {
			// Stored vectors are already prepared; initNode normalized
			// them again, which is a no-op up to rounding. Keep the stored
			// values exactly.
			copy(h.vecs.at(slot), ns.Embeddings)
		}
	}
	for id, ns := range s.Nodes {
		slot := h.ids[id]
		for layer, conns := range ns.Connections {
			blk := h.links(slot, layer)
			n := 0
			for _, nbID := range conns {
				nb, ok := h.ids[nbID]
				if !ok || n >= len(blk)-1 {
					continue
				}
				n++
				blk[n] = nb
			}
			blk[0] = int32(n)
		}
	}
	if ep, ok := h.ids[s.EntryPoint]; ok {
		h.setEntryPoint(ep, h.level(ep))
	} else {
		entry, maxLayer := int32(0), -1
		for _, slot := range h.ids {
			if lvl := h.level(slot); lvl > maxLayer {
				entry, maxLayer = slot, lvl
			}
		}
		h.setEntryPoint(entry, maxLayer)
	}

	if s.HasSQ {
		cb := &core.SQCodebook{Min: s.SQMin, Scale: s.SQScale, Dim: s.Dim}
		if !s.Normalized && h.metric.Normalizes() {
			// Older files trained SQ on raw vectors; retrain on the normalized ones.
			vecs := make([]core.Vector, 0, len(h.ids))
			for _, slot := range h.ids {
				vecs = append(vecs, core.Vector{Embeddings: h.vecs.at(slot)})
			}
			if c := core.NewSQCodebook(vecs); c != nil {
				cb = c
			}
		}
		h.setCodebook(cb)
	}
	return nil
}
