import { useEffect, useRef, useState } from 'react'
import { MOVIES, type Movie } from '../data.ts'
import { usePrefersReducedMotion } from '../hooks.ts'
import { Window } from './Window.tsx'

type Hit = { movie: Movie; distance: number }
type Point = { x: number; y: number }

const TOP_K = 3
const RANK_COLORS = ['#ff4fa3', '#ff9a3c', '#19d3c5']

function nearest(q: Point): Hit[] {
  return MOVIES.map(movie => ({ movie, distance: Math.hypot(movie.x - q.x, movie.y - q.y) }))
    .sort((a, b) => a.distance - b.distance)
    .slice(0, TOP_K)
}

const slug = (title: string) => title.toLowerCase().replace(/[^a-z]+/g, '-').replace(/-$/, '')

// The visitor's pointer is the query vector. The canvas redraws every frame;
// React state only changes when the result set does.
export function Playground() {
  const reduced = usePrefersReducedMotion()
  const mapRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const query = useRef<Point>({ x: 0.55, y: 0.42 })
  const target = useRef<Point>({ x: 0.55, y: 0.42 })
  const userMoved = useRef(false)
  const [touched, setTouched] = useState(false)
  const [result, setResult] = useState<{ q: Point; hits: Hit[] }>(() => ({ q: query.current, hits: nearest(query.current) }))

  useEffect(() => {
    const map = mapRef.current!, canvas = canvasRef.current!, ctx = canvas.getContext('2d')!
    let W = 0, H = 0, raf = 0, lastKey = ''

    const resize = () => {
      const r = map.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2)
      W = r.width; H = r.height
      canvas.width = W * dpr; canvas.height = H * dpr
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
    }
    const P = (x: number, y: number): [number, number] => [24 + x * (W - 48), 24 + y * (H - 64)]

    const draw = (t: number) => {
      const q = query.current, hits = nearest(q)
      ctx.clearRect(0, 0, W, H)
      ctx.strokeStyle = 'rgba(255,255,255,.04)'; ctx.lineWidth = 1
      for (let i = 1; i < 8; i++) { const x = (i * W) / 8; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke() }
      for (let i = 1; i < 6; i++) { const y = (i * H) / 6; ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke() }

      const [qx, qy] = P(q.x, q.y)
      hits.forEach((h, i) => {
        const [x, y] = P(h.movie.x, h.movie.y)
        ctx.strokeStyle = RANK_COLORS[i]; ctx.globalAlpha = 1 - i * 0.2; ctx.lineWidth = 2.2 - i * 0.4
        ctx.setLineDash([6, 6]); ctx.lineDashOffset = reduced ? 0 : -t / 40
        ctx.beginPath(); ctx.moveTo(qx, qy); ctx.lineTo(x, y); ctx.stroke()
      })
      ctx.setLineDash([]); ctx.globalAlpha = 1

      ctx.font = '500 13px "Instrument Sans", system-ui'
      for (const m of MOVIES) {
        const [x, y] = P(m.x, m.y), rank = hits.findIndex(h => h.movie === m)
        ctx.beginPath(); ctx.arc(x, y, rank >= 0 ? 7 : 5, 0, Math.PI * 2)
        ctx.fillStyle = rank >= 0 ? RANK_COLORS[rank] : '#3b4560'; ctx.fill()
        if (rank >= 0) { ctx.strokeStyle = 'rgba(255,255,255,.85)'; ctx.lineWidth = 2; ctx.stroke() }
        ctx.fillStyle = rank >= 0 ? '#f6f3ff' : '#7d88a3'
        const tw = ctx.measureText(m.title).width
        ctx.fillText(m.title, x + 11 + tw > W - 6 ? x - 11 - tw : x + 11, y + 4)
      }

      const pulse = reduced ? 0 : (Math.sin(t / 300) + 1) / 2
      ctx.beginPath(); ctx.arc(qx, qy, 16 + pulse * 6, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255,79,163,.16)'; ctx.fill()
      ctx.beginPath(); ctx.arc(qx, qy, 7, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill()

      const key = hits.map(h => h.movie.title + h.distance.toFixed(3)).join()
      if (key !== lastKey) { lastKey = key; setResult({ q: { ...q }, hits }) }
    }

    const loop = (t: number) => {
      if (!userMoved.current && !reduced) {
        // drift the query until the visitor takes over
        target.current = { x: 0.5 + 0.3 * Math.sin(t / 2600), y: 0.45 + 0.25 * Math.sin(t / 1900 + 1) }
      }
      const q = query.current, tg = target.current
      query.current = { x: q.x + (tg.x - q.x) * 0.12, y: q.y + (tg.y - q.y) * 0.12 }
      draw(t)
      raf = requestAnimationFrame(loop)
    }

    const point = (e: PointerEvent) => {
      const r = map.getBoundingClientRect()
      target.current = {
        x: Math.min(1, Math.max(0, (e.clientX - r.left - 24) / (r.width - 48))),
        y: Math.min(1, Math.max(0, (e.clientY - r.top - 24) / (r.height - 64))),
      }
      if (!userMoved.current) { userMoved.current = true; setTouched(true) }
    }

    const ro = new ResizeObserver(() => { resize(); draw(performance.now()) })
    ro.observe(map)
    map.addEventListener('pointermove', point)
    map.addEventListener('pointerdown', point)
    resize()
    raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(raf); ro.disconnect()
      map.removeEventListener('pointermove', point); map.removeEventListener('pointerdown', point)
    }
  }, [reduced])

  return (
    <div className="play">
      <Window title="Nearest-neighbour search over 12 movies">
        <div className="play-grid">
          <div className="map" ref={mapRef} role="img" aria-label="Map of movies placed by genre. Move the pointer to search for the closest ones.">
            <canvas ref={canvasRef} />
            <div className="hint" style={{ opacity: touched ? 0 : 1 }}>Move your pointer over the map. It is the query vector.</div>
          </div>
          <SearchJSON q={result.q} hits={result.hits} />
        </div>
      </Window>
      <p className="caption">
        Each movie is a vector of genre scores. goatdb ranks them by distance to your query and returns the top 3 with
        their metadata. Real embeddings have hundreds of dimensions; the search works the same way.
      </p>
    </div>
  )
}

const K = ({ children }: { children: string }) => <span className="k">"{children}"</span>
const S = ({ children }: { children: string }) => <span className="s">"{children}"</span>
const N = ({ children }: { children: string | number }) => <span className="n">{children}</span>

function SearchJSON({ q, hits }: { q: Point; hits: Hit[] }) {
  return (
    <div className="json" aria-live="polite">
      <span className="c">POST /collections/movies/search</span>{'\n'}
      {'{ '}<K>embeddings</K>: <N>{`[${q.x.toFixed(2)}, ${q.y.toFixed(2)}]`}</N>, <K>top_k</K>: <N>{TOP_K}</N>{' }'}{'\n\n'}
      <span className="c">200 OK</span>{'\n'}
      {'{\n  '}<K>results</K>{': [\n'}
      {hits.map((h, i) => (
        <span key={h.movie.title}>
          {'    {\n      '}<K>id</K>: <S>{slug(h.movie.title)}</S>{',\n      '}
          <K>distance</K>: <N>{h.distance.toFixed(4)}</N>{',\n      '}
          <K>metadata</K>{': { '}<K>title</K>: <S>{h.movie.title}</S>{' }\n    }'}{i < hits.length - 1 ? ',' : ''}{'\n'}
        </span>
      ))}
      {'  ]\n}'}
    </div>
  )
}
