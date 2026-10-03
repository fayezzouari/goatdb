package storage

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"

	"golang.org/x/sys/unix"
)

const deletedFlag = byte(1)

type VectorStore struct {
	f          *os.File
	data       []byte
	dim        int
	recordSize int
	capacity   int
}

func openVectorStore(path string, dim int) (*VectorStore, error) {
	recordSize := dim*4 + 1
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}

	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}

	capacity := 1024
	size := info.Size()
	if size == 0 {
		size = int64(recordSize * capacity)
		if err := f.Truncate(size); err != nil {
			f.Close()
			return nil, err
		}
	} else {
		capacity = int(size) / recordSize
	}

	data, err := syscall.Mmap(int(f.Fd()), 0, int(size),
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		f.Close()
		return nil, err
	}

	return &VectorStore{
		f:          f,
		data:       data,
		dim:        dim,
		recordSize: recordSize,
		capacity:   capacity,
	}, nil
}

func (vs *VectorStore) grow(minSlot int) error {
	newCapacity := vs.capacity
	for newCapacity <= minSlot {
		newCapacity *= 2
	}
	newSize := int64(newCapacity * vs.recordSize)
	if err := syscall.Munmap(vs.data); err != nil {
		return err
	}
	if err := vs.f.Truncate(newSize); err != nil {
		return err
	}
	data, err := syscall.Mmap(int(vs.f.Fd()), 0, int(newSize),
		syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED)
	if err != nil {
		return err
	}
	vs.data = data
	vs.capacity = newCapacity
	return nil
}

func (vs *VectorStore) Write(slot int, embeddings []float32) error {
	if len(embeddings) != vs.dim {
		return fmt.Errorf("vector store: expected %d dimensions, got %d", vs.dim, len(embeddings))
	}
	if slot >= vs.capacity {
		if err := vs.grow(slot); err != nil {
			return err
		}
	}
	off := slot * vs.recordSize
	rec := vs.data[off : off+vs.recordSize]
	for i, v := range embeddings {
		binary.LittleEndian.PutUint32(rec[i*4:], math.Float32bits(v))
	}
	rec[vs.dim*4] = 0
	return nil
}

func (vs *VectorStore) Read(slot int) ([]float32, bool) {
	if slot >= vs.capacity {
		return nil, false
	}
	off := slot * vs.recordSize
	rec := vs.data[off : off+vs.recordSize]
	if rec[vs.dim*4] == deletedFlag {
		return nil, false
	}
	emb := make([]float32, vs.dim)
	for i := range emb {
		emb[i] = math.Float32frombits(binary.LittleEndian.Uint32(rec[i*4:]))
	}
	return emb, true
}

func (vs *VectorStore) Delete(slot int) {
	if slot < vs.capacity {
		vs.data[slot*vs.recordSize+vs.dim*4] = deletedFlag
	}
}

// Flush writes dirty pages of the mapping and the file size to disk.
func (vs *VectorStore) Flush() error {
	if err := unix.Msync(vs.data, unix.MS_SYNC); err != nil {
		return err
	}
	return vs.f.Sync()
}

func (vs *VectorStore) Close() error {
	flushErr := vs.Flush()
	syscall.Munmap(vs.data)
	if err := vs.f.Close(); err != nil {
		return err
	}
	return flushErr
}
