export const REPO_URL = 'https://github.com/fayezzouari/goatdb'
export const YOUTUBE_ID = 'pu9x9_9ALDQ'
export const INSTALL_CMD = 'go install github.com/fayezzouari/goatdb/cmd/goatdb@latest'

export type Movie = { title: string; year: number; x: number; y: number }

// A 2-D layout of the demo dataset in genre space:
// x runs from drama/romance to sci-fi/action, y from light to dark.
export const MOVIES: Movie[] = [
  { title: 'Interstellar', year: 2014, x: 0.78, y: 0.38 },
  { title: 'Alien', year: 1979, x: 0.8, y: 0.8 },
  { title: 'Galaxy Quest', year: 1999, x: 0.7, y: 0.18 },
  { title: "Hitchhiker's Guide", year: 2005, x: 0.62, y: 0.12 },
  { title: 'WALL-E', year: 2008, x: 0.5, y: 0.26 },
  { title: 'The Notebook', year: 2004, x: 0.1, y: 0.45 },
  { title: 'La La Land', year: 2016, x: 0.16, y: 0.25 },
  { title: 'Mad Max: Fury Road', year: 2015, x: 0.9, y: 0.58 },
  { title: 'Toy Story', year: 1995, x: 0.38, y: 0.14 },
  { title: 'Get Out', year: 2017, x: 0.36, y: 0.86 },
  { title: 'Planet Earth', year: 2006, x: 0.14, y: 0.74 },
  { title: 'Die Hard', year: 1988, x: 0.66, y: 0.64 },
]

// SIFT-1M, HNSW M=16 efConstruction=200, measured for #35/#36.
export type BenchPoint = { ef: number; recall: number; qps: number }
export const SIFT_1M: BenchPoint[] = [
  { ef: 16, recall: 0.8048, qps: 2245 },
  { ef: 32, recall: 0.9042, qps: 1988 },
  { ef: 64, recall: 0.9657, qps: 1713 },
  { ef: 128, recall: 0.9887, qps: 1448 },
  { ef: 256, recall: 0.9978, qps: 996 },
  { ef: 512, recall: 0.9988, qps: 724 },
]
