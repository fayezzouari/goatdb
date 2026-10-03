package storage

import (
	"bufio"
	"encoding/binary"
	"errors"
	"fmt"
	"hash/crc32"
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

// Each record is framed as [payload length u32][crc32c of payload u32][payload].
// Payload: op u8, id length u32, id, slot u32, embedding count u32, embeddings.
const (
	walHeaderSize   = 8
	walPayloadFixed = 1 + 4 + 4 + 4
	walMaxIDLen     = 1 << 16
	walMaxDim       = 1 << 16
	walMaxPayload   = walPayloadFixed + walMaxIDLen + walMaxDim*4
)

var crcTable = crc32.MakeTable(crc32.Castagnoli)

var errTornRecord = errors.New("wal: torn or corrupt record")

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
	entries, valid, err := replayWAL(f)
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, nil, err
	}
	if info.Size() > valid {
		if err := f.Truncate(valid); err != nil {
			f.Close()
			return nil, nil, err
		}
		if err := f.Sync(); err != nil {
			f.Close()
			return nil, nil, err
		}
	}
	if _, err := f.Seek(valid, io.SeekStart); err != nil {
		f.Close()
		return nil, nil, err
	}
	return &WAL{f: f, size: valid}, entries, nil
}

// replayWAL decodes records until EOF or the first torn or corrupt record,
// and returns the decoded entries and the length of the valid prefix.
func replayWAL(f *os.File) ([]walEntry, int64, error) {
	if _, err := f.Seek(0, io.SeekStart); err != nil {
		return nil, 0, err
	}
	r := bufio.NewReaderSize(f, 1<<16)
	var (
		entries []walEntry
		valid   int64
		header  [walHeaderSize]byte
		payload []byte
	)
	for {
		if _, err := io.ReadFull(r, header[:]); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return entries, valid, nil
			}
			return nil, 0, err
		}
		n := binary.LittleEndian.Uint32(header[0:])
		sum := binary.LittleEndian.Uint32(header[4:])
		if n < walPayloadFixed || n > walMaxPayload {
			return entries, valid, nil
		}
		if cap(payload) < int(n) {
			payload = make([]byte, n)
		}
		payload = payload[:n]
		if _, err := io.ReadFull(r, payload); err != nil {
			if errors.Is(err, io.EOF) || errors.Is(err, io.ErrUnexpectedEOF) {
				return entries, valid, nil
			}
			return nil, 0, err
		}
		if crc32.Checksum(payload, crcTable) != sum {
			return entries, valid, nil
		}
		e, err := decodeRecord(payload)
		if err != nil {
			return entries, valid, nil
		}
		entries = append(entries, e)
		valid += walHeaderSize + int64(n)
	}
}

func decodeRecord(p []byte) (walEntry, error) {
	op := opType(p[0])
	if op != opInsert && op != opDelete && op != opUpdate {
		return walEntry{}, errTornRecord
	}
	idLen := binary.LittleEndian.Uint32(p[1:])
	if idLen > walMaxIDLen || int(idLen) > len(p)-walPayloadFixed {
		return walEntry{}, errTornRecord
	}
	off := 5
	id := string(p[off : off+int(idLen)])
	off += int(idLen)
	slot := binary.LittleEndian.Uint32(p[off:])
	off += 4
	embLen := binary.LittleEndian.Uint32(p[off:])
	off += 4
	if embLen > walMaxDim || int(embLen)*4 != len(p)-off {
		return walEntry{}, errTornRecord
	}
	var embeddings []float32
	if embLen > 0 {
		embeddings = make([]float32, embLen)
		for i := range embeddings {
			embeddings[i] = math.Float32frombits(binary.LittleEndian.Uint32(p[off+i*4:]))
		}
	}
	return walEntry{op: op, id: id, slot: slot, embeddings: embeddings}, nil
}

func (w *WAL) Append(op opType, id string, slot uint32, embeddings []float32) error {
	if err := checkRecord(id, embeddings); err != nil {
		return err
	}
	buf := make([]byte, 0, recordSize(id, embeddings))
	return w.write(appendRecord(buf, op, id, slot, embeddings))
}

// AppendBatch encodes all entries into a single write.
func (w *WAL) AppendBatch(entries []walEntry) error {
	size := 0
	for _, e := range entries {
		if err := checkRecord(e.id, e.embeddings); err != nil {
			return err
		}
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

func checkRecord(id string, embeddings []float32) error {
	if len(id) > walMaxIDLen {
		return fmt.Errorf("wal: id length %d exceeds %d", len(id), walMaxIDLen)
	}
	if len(embeddings) > walMaxDim {
		return fmt.Errorf("wal: %d dimensions exceeds %d", len(embeddings), walMaxDim)
	}
	return nil
}

func recordSize(id string, embeddings []float32) int {
	return walHeaderSize + walPayloadFixed + len(id) + len(embeddings)*4
}

func appendRecord(buf []byte, op opType, id string, slot uint32, embeddings []float32) []byte {
	start := len(buf)
	buf = append(buf, make([]byte, walHeaderSize)...)
	buf = append(buf, byte(op))
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(id)))
	buf = append(buf, id...)
	buf = binary.LittleEndian.AppendUint32(buf, slot)
	buf = binary.LittleEndian.AppendUint32(buf, uint32(len(embeddings)))
	for _, v := range embeddings {
		buf = binary.LittleEndian.AppendUint32(buf, math.Float32bits(v))
	}
	payload := buf[start+walHeaderSize:]
	binary.LittleEndian.PutUint32(buf[start:], uint32(len(payload)))
	binary.LittleEndian.PutUint32(buf[start+4:], crc32.Checksum(payload, crcTable))
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
