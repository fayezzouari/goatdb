package db

import (
	"errors"
	"fmt"
)

// ErrInvalidParams is returned when index or search parameters are out of range.
var ErrInvalidParams = errors.New("invalid parameters")

// Default index parameters, used when a collection is created without them
// and when loading a config.json written before parameters were stored.
const (
	DefaultHNSWM              = 16
	DefaultHNSWEfConstruction = 200
	DefaultHNSWEfSearch       = 128
	DefaultIVFNList           = 100
	DefaultIVFNProbe          = 20

	defaultLSHTables = 20
	defaultLSHBits   = 8
)

// Upper bounds for index and search parameters.
const (
	MaxHNSWM   = 512
	MaxEf      = 10000
	MaxIVFList = 65536
)

// HNSWParams configures an HNSW index. Zero fields take their defaults.
type HNSWParams struct {
	M              int `json:"m"`
	EfConstruction int `json:"ef_construction"`
	EfSearch       int `json:"ef_search"`
}

// IVFParams configures an IVF index. Zero fields take their defaults.
type IVFParams struct {
	NList  int `json:"nlist"`
	NProbe int `json:"nprobe"`
}

// CollectionOptions holds optional index parameters for CreateCollectionWithOptions.
// Only the parameters matching the collection's index type are used.
type CollectionOptions struct {
	HNSW *HNSWParams
	IVF  *IVFParams
}

func resolveHNSW(p *HNSWParams) (HNSWParams, error) {
	out := HNSWParams{M: DefaultHNSWM, EfConstruction: DefaultHNSWEfConstruction, EfSearch: DefaultHNSWEfSearch}
	if p != nil {
		if p.M != 0 {
			out.M = p.M
		}
		if p.EfConstruction != 0 {
			out.EfConstruction = p.EfConstruction
		}
		if p.EfSearch != 0 {
			out.EfSearch = p.EfSearch
		}
	}
	// M must be at least 2: the level multiplier is 1/ln(M).
	if out.M < 2 || out.M > MaxHNSWM {
		return out, fmt.Errorf("%w: hnsw.m must be between 2 and %d", ErrInvalidParams, MaxHNSWM)
	}
	if out.EfConstruction < 1 || out.EfConstruction > MaxEf {
		return out, fmt.Errorf("%w: hnsw.ef_construction must be between 1 and %d", ErrInvalidParams, MaxEf)
	}
	if out.EfSearch < 1 || out.EfSearch > MaxEf {
		return out, fmt.Errorf("%w: hnsw.ef_search must be between 1 and %d", ErrInvalidParams, MaxEf)
	}
	return out, nil
}

func resolveIVF(p *IVFParams) (IVFParams, error) {
	out := IVFParams{NList: DefaultIVFNList, NProbe: DefaultIVFNProbe}
	if p != nil {
		if p.NList != 0 {
			out.NList = p.NList
		}
		if p.NProbe != 0 {
			out.NProbe = p.NProbe
		}
	}
	if out.NList < 1 || out.NList > MaxIVFList {
		return out, fmt.Errorf("%w: ivf.nlist must be between 1 and %d", ErrInvalidParams, MaxIVFList)
	}
	if out.NProbe < 1 || out.NProbe > out.NList {
		return out, fmt.Errorf("%w: ivf.nprobe must be between 1 and nlist (%d)", ErrInvalidParams, out.NList)
	}
	return out, nil
}

// resolveOptions fills defaults for the parameters of indexType, validates
// them, and drops parameters that do not apply to indexType.
func resolveOptions(indexType string, opts CollectionOptions) (CollectionOptions, error) {
	var out CollectionOptions
	switch indexType {
	case "hnsw":
		p, err := resolveHNSW(opts.HNSW)
		if err != nil {
			return out, err
		}
		out.HNSW = &p
	case "ivf":
		p, err := resolveIVF(opts.IVF)
		if err != nil {
			return out, err
		}
		out.IVF = &p
	}
	return out, nil
}

// ValidateEf reports whether ef is a valid per-query search depth.
func ValidateEf(ef int) error {
	if ef < 1 || ef > MaxEf {
		return fmt.Errorf("%w: ef must be between 1 and %d", ErrInvalidParams, MaxEf)
	}
	return nil
}
