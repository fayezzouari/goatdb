package storage

import (
	"encoding/binary"
	"hash/crc32"
	"os"
	"testing"
)

func writeTestWAL(t *testing.T, path string, ids ...string) []int64 {
	t.Helper()
	wal, _, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	var ends []int64
	for i, id := range ids {
		if err := wal.Append(opInsert, id, uint32(i), []float32{float32(i), 1, 2}); err != nil {
			t.Fatal(err)
		}
		ends = append(ends, wal.Size())
	}
	wal.Sync()
	wal.Close()
	return ends
}

func fileSize(t *testing.T, path string) int64 {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatal(err)
	}
	return info.Size()
}

func patchFile(t *testing.T, path string, off int64, b []byte) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_RDWR, 0644)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteAt(b, off); err != nil {
		t.Fatal(err)
	}
}

func TestWALTornTail(t *testing.T) {
	for _, cut := range []int64{1, 4, 8, 9, 20} {
		path := t.TempDir() + "/wal.log"
		ends := writeTestWAL(t, path, "a", "b", "c")
		if err := os.Truncate(path, ends[2]-cut); err != nil {
			t.Fatal(err)
		}

		wal, entries, err := openWAL(path)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 2 || entries[0].id != "a" || entries[1].id != "b" {
			t.Fatalf("cut %d: expected [a b], got %v", cut, entries)
		}
		if got := fileSize(t, path); got != ends[1] {
			t.Errorf("cut %d: expected torn tail truncated to %d, got %d", cut, ends[1], got)
		}

		if err := wal.Append(opInsert, "d", 9, []float32{4, 5, 6}); err != nil {
			t.Fatal(err)
		}
		wal.Sync()
		wal.Close()

		_, entries, err = openWAL(path)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 3 || entries[2].id != "d" || entries[2].embeddings[2] != 6 {
			t.Errorf("cut %d: expected append after truncation to replay, got %v", cut, entries)
		}
	}
}

func TestWALChecksumMismatch(t *testing.T) {
	path := t.TempDir() + "/wal.log"
	ends := writeTestWAL(t, path, "a", "b", "c")
	patchFile(t, path, ends[2]-2, []byte{0xff})

	_, entries, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 2 {
		t.Fatalf("expected 2 entries, got %d", len(entries))
	}
	if got := fileSize(t, path); got != ends[1] {
		t.Errorf("expected corrupt record truncated, size %d want %d", got, ends[1])
	}
}

func TestWALCorruptRecordStopsReplay(t *testing.T) {
	path := t.TempDir() + "/wal.log"
	ends := writeTestWAL(t, path, "a", "b", "c")
	patchFile(t, path, ends[0]+walHeaderSize+1, []byte{0xff})

	_, entries, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 1 || entries[0].id != "a" {
		t.Fatalf("expected replay to stop after a, got %v", entries)
	}
}

func TestWALCorruptLength(t *testing.T) {
	for _, n := range []uint32{0, walPayloadFixed - 1, walMaxPayload + 1, 1<<32 - 1} {
		path := t.TempDir() + "/wal.log"
		ends := writeTestWAL(t, path, "a", "b")
		var buf [4]byte
		binary.LittleEndian.PutUint32(buf[:], n)
		patchFile(t, path, ends[0], buf[:])

		_, entries, err := openWAL(path)
		if err != nil {
			t.Fatal(err)
		}
		if len(entries) != 1 {
			t.Errorf("length %d: expected 1 entry, got %d", n, len(entries))
		}
	}
}

func TestWALInnerLengthMismatch(t *testing.T) {
	path := t.TempDir() + "/wal.log"
	wal, _, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	rec := appendRecord(nil, opInsert, "a", 0, []float32{1, 2})
	// claim more embeddings than the payload holds, with a valid checksum
	payload := rec[walHeaderSize:]
	binary.LittleEndian.PutUint32(payload[1+4+1+4:], 1000)
	binary.LittleEndian.PutUint32(rec[4:], crc32.Checksum(payload, crcTable))
	wal.write(rec)
	wal.Close()

	_, entries, err := openWAL(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 0 {
		t.Errorf("expected inconsistent record to be rejected, got %v", entries)
	}
}

func TestWALRejectsOversizedRecord(t *testing.T) {
	wal, _, err := openWAL(t.TempDir() + "/wal.log")
	if err != nil {
		t.Fatal(err)
	}
	defer wal.Close()
	if err := wal.Append(opInsert, "a", 0, make([]float32, walMaxDim+1)); err == nil {
		t.Error("expected error for oversized embeddings")
	}
	if wal.Size() != 0 {
		t.Errorf("expected nothing written, got %d bytes", wal.Size())
	}
}

func TestStoreTornWALTailIgnored(t *testing.T) {
	dir := t.TempDir()
	s, err := Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	s.Add("a", []float32{1, 2}, nil)
	slot, _ := s.meta.GetSlot("a")
	rec := appendRecord(nil, opUpdate, "a", slot, []float32{7, 8})
	s.wal.f.Write(rec[:len(rec)-3])
	crash(t, s)

	s, err = Open(dir, 2)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	emb, _, err := s.Get("a")
	if err != nil {
		t.Fatal(err)
	}
	if emb[0] != 1 || emb[1] != 2 {
		t.Errorf("torn record was applied: got %v", emb)
	}
}
