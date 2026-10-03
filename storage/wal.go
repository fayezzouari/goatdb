package storage

import (
	"encoding/binary"
	"io"
	"math"
	"os"
)

type opType uint8

const (
	opInsert opType = 1
	opDelete opType = 2
	opUpdate opType = 3
)

type walEntry struct {
	op         opType
	id         string
	slot       uint32
	embeddings []float32
}

type WAL struct {
	f    *os.File
	size int64
}

func openWAL(path string) (*WAL, []walEntry, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, nil, err
	}
	entries, _ := replayWAL(f)
	size, err := f.Seek(0, io.SeekEnd)
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	return &WAL{f: f, size: size}, entries, nil
}

func replayWAL(f *os.File) ([]walEntry, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, err
	}
	var entries []walEntry
	for {
		var op uint8
		if err := binary.Read(f, binary.LittleEndian, &op); err != nil {
			break
		}
		var idLen uint32
		if err := binary.Read(f, binary.LittleEndian, &idLen); err != nil {
			break
		}
		idBytes := make([]byte, idLen)
		if _, err := io.ReadFull(f, idBytes); err != nil {
			break
		}
		var slot uint32
		if err := binary.Read(f, binary.LittleEndian, &slot); err != nil {
			break
		}
		var embLen uint32
		if err := binary.Read(f, binary.LittleEndian, &embLen); err != nil {
			break
		}
		embeddings := make([]float32, embLen)
		for i := range embeddings {
			var bits uint32
			if err := binary.Read(f, binary.LittleEndian, &bits); err != nil {
				break
			}
			embeddings[i] = math.Float32frombits(bits)
		}
		entries = append(entries, walEntry{
			op:         opType(op),
			id:         string(idBytes),
			slot:       slot,
			embeddings: embeddings,
		})
	}
	return entries, nil
}

func (w *WAL) Append(op opType, id string, slot uint32, embeddings []float32) error {
	buf := make([]byte, 0, recordSize(id, embeddings))
	return w.write(appendRecord(buf, op, id, slot, embeddings))
}

// AppendBatch encodes all entries into a single write.
func (w *WAL) AppendBatch(entries []walEntry) error {
	size := 0
	for _, e := range entries {
		size += recordSize(e.id, e.embeddings)
	}
	buf := make([]byte, 0, size)
	for _, e := range entries {
		buf = appendRecord(buf, e.op, e.id, e.slot, e.embeddings)
	}
	return w.write(buf)
}

func (w *WAL) write(buf []byte) error {
	if _, err := w.f.Write(buf); err != nil {
		// drop the partial write so later records stay aligned
		w.f.Truncate(w.size)
		w.f.Seek(w.size, io.SeekStart)
		return err
	}
	w.size += int64(len(buf))
	return nil
}

func (w *WAL) Size() int64 {
	return w.size
}

func recordSize(id string, embeddings []float32) int {
	return 1 + 4 + len(id) + 4 + 4 + len(embeddings)*4
}

func appendRecord(buf []byte, op opType, id string, slot uint32, embeddings []float32) []byte {
	buf = append(buf, byte(op))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(id)))
	buf = append(buf, id...)
	buf = binary.LittleEndian.AppendUint32(buf, slot)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(embeddings)))
	for _, v := range embeddings {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
	}
	return buf
}

func (w *WAL) Sync() error {
	return w.f.Sync()
}

func (w *WAL) Truncate() error {
	if err := w.f.Truncate(0); err != nil {
		return err
	}
	if _, err := w.f.Seek(0, io.SeekStart); err != nil {
		return err
	}
	w.size = 0
	return w.f.Sync()
}

func (w *WAL) Close() error {
	return w.f.Close()
}
