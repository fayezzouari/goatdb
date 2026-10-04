import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties } from 'react'
import { SIFT_1M } from '../data.ts'

const X0 = 56, X1 = 540, Y0 = 290, Y1 = 20
const rx = (r: number) => X0 + ((r - 0.78) / (1.0 - 0.78)) * (X1 - X0)
const qy = (v: number) => Y0 - ((v - 500) / (2500 - 500)) * (Y0 - Y1)
const PATH = SIFT_1M.map((p, i) => `${i ? 'L' : 'M'}${rx(p.recall).toFixed(1)} ${qy(p.qps).toFixed(1)}`).join(' ')

function Chart() {
  const box = useRef<HTMLDivElement>(null)
  const curve = useRef<SVGPathElement>(null)
  const [len, setLen] = useState(2000)
  const [drawn, setDrawn] = useState(false)

  useLayoutEffect(() => setLen(curve.current!.getTotalLength()), [])
  // draw the curve once, when it first scrolls into view
  useEffect(() => {
    const io = new IntersectionObserver(es => {
      if (es.some(e => e.isIntersecting)) { setDrawn(true); io.disconnect() }
    }, { threshold: 0.4 })
    io.observe(box.current!)
    return () => io.disconnect()
  }, [])

  return (
    <div ref={box} className={`window chart ${drawn ? 'drawn' : ''}`}>
        <svg viewBox="0 0 560 330" role="img" aria-label="Recall at 10 against queries per second for search depth 16 to 512">
          <defs>
            <linearGradient id="grad" x1="0" x2="1">
              <stop offset="0" stopColor="#ff9a3c" /><stop offset=".5" stopColor="#ff4fa3" /><stop offset="1" stopColor="#6a5cff" />
            </linearGradient>
          </defs>
          {[500, 1000, 1500, 2000, 2500].map(v => (
            <g key={v}>
              <line x1={X0} x2={X1} y1={qy(v)} y2={qy(v)} className="grid" />
              <text x={X0 - 10} y={qy(v) + 4} textAnchor="end">{v.toLocaleString('en')}</text>
            </g>
          ))}
          {[0.8, 0.85, 0.9, 0.95, 1.0].map(r => (
            <text key={r} x={rx(r)} y={Y0 + 22} textAnchor="middle">{r.toFixed(2)}</text>
          ))}
          <line x1={X0} x2={X1} y1={Y0} y2={Y0} className="axis" />
          <text x={(X0 + X1) / 2} y={Y0 + 40} textAnchor="middle">recall@10</text>
          <text x={14} y={(Y0 + Y1) / 2} transform={`rotate(-90 14 ${(Y0 + Y1) / 2})`} textAnchor="middle">queries per second</text>
          <path ref={curve} d={PATH} className="curve" style={{ '--len': len } as CSSProperties} />
          {SIFT_1M.map((p, i) => (
            <g key={p.ef} style={{ '--i': i } as CSSProperties}>
              <circle cx={rx(p.recall)} cy={qy(p.qps)} r={5.5} className="pt" />
              <text x={rx(p.recall) + (i > 3 ? -10 : 10)} y={qy(p.qps) - 12} className="lbl" textAnchor={i > 3 ? 'end' : 'start'}>ef {p.ef}</text>
            </g>
          ))}
        </svg>
    </div>
  )
}

export function Benchmarks() {
  return (
    <section id="benchmarks" className="wrap">
      <h2>Measured on SIFT-1M</h2>
      <p className="intro">
        One million 128-dimension vectors from the standard ann-benchmarks set, HNSW with M=16 and efConstruction=200,
        1,000 queries over HTTP.
      </p>
      <div className="bench">
        <Chart />
        <div className="stats">
          <div className="stat"><div className="big">0.99<small>recall@10</small></div><p>at 0.71 ms median latency, ef=128</p></div>
          <div className="stat"><div className="big">85<small>s</small></div><p>to insert and index 1M vectors, 5&times; faster than the single-threaded build</p></div>
          <div className="stat"><div className="big">1.7<small>GB</small></div><p>peak memory for the 1M index, down from 4.4 GB</p></div>
        </div>
      </div>
      <p className="fine">
        Apple M4 Pro, goatdb limited to 8 cores, Python client over HTTP/JSON with keep-alive; latency includes the HTTP
        round trip. Reproduce it with <code>bench/compare</code> in the repository, which also runs Qdrant, Weaviate,
        Milvus, pgvector and Chroma under the same limits.
      </p>
    </section>
  )
}
