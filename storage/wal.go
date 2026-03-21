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
	f *os.File
}

func openWAL(path string) (*WAL, []walEntry, error) {
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, nil, err
	}
	entries, _ := replayWAL(f)
	return &WAL{f: f}, entries, nil
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
	size := 1 + 4 + len(id) + 4 + 4 + len(embeddings)*4
	buf := make([]byte, size)
	off := 0

	buf[off] = byte(op)
	off++
	binary.LittleEndian.PutUint32(buf[off:], uint32(len(id)))
	off += 4
	copy(buf[off:], id)
	off += len(id)
	binary.LittleEndian.PutUint32(buf[off:], slot)
	off += 4
	binary.LittleEndian.PutUint32(buf[off:], uint32(len(embeddings)))
	off += 4
	for _, v := range embeddings {
		binary.LittleEndian.PutUint32(buf[off:], math.Float32bits(v))
		off += 4
	}

	_, err := w.f.Write(buf)
	return err
}

func (w *WAL) Sync() error {
	return w.f.Sync()
}

func (w *WAL) Truncate() error {
	if err := w.f.Truncate(0); err != nil {
		return err
	}
	_, err := w.f.Seek(0, io.SeekStart)
	return err
}

func (w *WAL) Close() error {
	return w.f.Close()
}
