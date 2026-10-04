import { useEffect, useRef } from 'react'
import { usePrefersReducedMotion } from '../hooks.ts'

const PX = 6 // css pixels per art pixel

// 4x4 ordered (Bayer) dither thresholds
const BAYER = [0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5].map(v => (v + 0.5) / 16)
const dither = (x: number, y: number) => BAYER[(y & 3) * 4 + (x & 3)]

const hex = (h: string) => {
  const n = parseInt(h.slice(1), 16)
  return (255 << 24) | ((n & 255) << 16) | (((n >> 8) & 255) << 8) | (n >> 16) // ABGR for ImageData
}
const C = {
  space: hex('#060608'), spaceLow: hex('#0b0b0e'), sky: hex('#111115'), skyLow: hex('#18181d'),
  starDim: hex('#34343b'), star: hex('#6c6c76'), starHot: hex('#a9a9b3'),
  moon: hex('#3f3f46'), moonShade: hex('#26262c'),
  cloudTop: hex('#323239'), cloud: hex('#24242a'), cloudLow: hex('#1b1b20'),
  far: hex('#1a1a1f'), snow: hex('#33333a'), mid: hex('#202025'), midEdge: hex('#29292f'),
  near: hex('#26262b'), nearEdge: hex('#303036'),
  grass: hex('#3a3a40'), soil: hex('#1c1c20'), soilDark: hex('#141417'), pebble: hex('#2c2c31'),
}

// Integer hash -> [0, 1)
function hash(x: number, y: number, s = 0) {
  let h = (x * 374761393 + y * 668265263 + s * 2147483647) | 0
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}
// Smooth 2-D value noise
function noise(x: number, y: number, s: number) {
  const xi = Math.floor(x), yi = Math.floor(y), xf = x - xi, yf = y - yi
  const u = xf * xf * (3 - 2 * xf), v = yf * yf * (3 - 2 * yf)
  const a = hash(xi, yi, s), b = hash(xi + 1, yi, s), c = hash(xi, yi + 1, s), d = hash(xi + 1, yi + 1, s)
  return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v
}
const ridge = (x: number, f: number, s: number) =>
  (Math.sin(x * f + s) + Math.sin(x * f * 2.3 + s * 3) * 0.5 + Math.sin(x * f * 5.1 + s * 7) * 0.22) / 1.72

// A tall pixel world behind the page: space at the top, clouds in the middle,
// mountains below and the ground at the footer, scrolled with the page.
export function Background() {
  const ref = useRef<HTMLCanvasElement>(null)
  const reduced = usePrefersReducedMotion()

  useEffect(() => {
    const canvas = ref.current!, ctx = canvas.getContext('2d')!
    let W = 0, H = 0, ground = 1000, img: ImageData, buf: Uint32Array, raf = 0, last = 0

    function measure() {
      W = Math.ceil(innerWidth / PX); H = Math.ceil(innerHeight / PX) + 1
      canvas.width = W; canvas.height = H
      canvas.style.height = `${H * PX}px`
      img = ctx.createImageData(W, H)
      buf = new Uint32Array(img.data.buffer)
      // the ground starts at the footer rule, so the goat stands on it
      const rule = document.querySelector('.foot-row')
      const top = rule ? rule.getBoundingClientRect().top + scrollY : document.body.scrollHeight - 200
      ground = Math.round(top / PX)
    }

    function color(x: number, y: number, t: number) {
      const g = ground

      // ground
      if (y >= g) {
        const d = y - g
        if (d === 0) return hash(x, 1, 9) < 0.15 ? C.soil : C.grass
        if (d === 1) return (x & 1) ? C.grass : C.soil
        if (hash(x, y, 4) < 0.03) return C.pebble
        return d > 6 && dither(x, y) < 0.5 ? C.soilDark : C.soil
      }

      // mountains, near to far, only in the last stretch above the ground
      const near = g - 16 - Math.round(7 * ridge(x, 0.05, 2))
      if (y >= near) return y === near ? C.nearEdge : C.near
      const mid = g - 34 - Math.round(12 * ridge(x, 0.03, 5))
      if (y >= mid) return y === mid ? C.midEdge : C.mid
      const far = g - 58 - Math.round(20 * ridge(x, 0.017, 11))
      if (y >= far) {
        const peak = far < g - 66
        return peak && y - far < 3 && (y - far === 0 || ((x + y) & 1) === 0) ? C.snow : C.far
      }

      // altitude: 0 at the top of the page, 1 at the ground
      const a = y / g

      // clouds in three drifting bands through the middle of the descent
      for (const [center, speed, scale, seed] of [[0.4, 0.6, 22, 1], [0.55, 1, 16, 2], [0.7, 1.5, 12, 3]]) {
        const band = 1 - Math.abs(a - center) / 0.09
        if (band <= 0) continue
        const nx = (x + t * speed) / scale, ny = y / (scale * 0.45)
        const n = noise(nx, ny, seed) * 0.65 + noise(nx * 2.1, ny * 2.1, seed + 9) * 0.35
        const edge = n * band - 0.42
        if (edge > 0) {
          const above = noise(nx, (y - 1) / (scale * 0.45), seed) * 0.65 + noise(nx * 2.1, ((y - 1) / (scale * 0.45)) * 2.1, seed + 9) * 0.35
          if (above * band - 0.42 <= 0) return C.cloudTop
          return edge < 0.05 && dither(x, y) < 0.5 ? C.cloudLow : C.cloud
        }
      }

      // moon near the top
      const mx = Math.round(W * 0.935), my = 24, mr = 7
      const md = Math.hypot(x - mx, y - my)
      if (md <= mr) return (x - mx + mr) / (2 * mr) > dither(x, y) * 0.8 + 0.3 ? C.moon : C.moonShade

      // stars thin out as the sky gets lighter
      const density = Math.max(0, 1 - a / 0.5)
      const h = hash(x, y, 1)
      if (h < 0.006 * density) {
        const tw = reduced ? 0.5 : (Math.sin(t * 0.08 + h * 9000) + 1) / 2
        return h < 0.0012 * density ? (tw > 0.3 ? C.starHot : C.star) : tw > 0.5 ? C.star : C.starDim
      }

      // dithered sky: black space at the top, lighter grey near the ground
      if (a < 0.45) return a / 0.45 > dither(x, y) ? C.spaceLow : C.space
      const s = (a - 0.45) / 0.55
      if (s < 0.35) return s / 0.35 > dither(x, y) ? C.sky : C.spaceLow
      return (s - 0.35) / 0.65 > dither(x, y) ? C.skyLow : C.sky
    }

    function draw(time: number) {
      const t = reduced ? 0 : time / 160
      const top = Math.floor(scrollY / PX)
      canvas.style.transform = `translateY(${-(scrollY - top * PX)}px)`
      for (let vy = 0; vy < H; vy++) {
        const y = top + vy, row = vy * W
        for (let x = 0; x < W; x++) buf[row + x] = color(x, y, t)
      }
      ctx.putImageData(img, 0, 0)
    }

    function loop(time: number) {
      raf = requestAnimationFrame(loop)
      if (time - last < 66) return // ~15 fps is plenty for drifting pixels
      last = time
      draw(time)
    }
    const onScroll = () => draw(performance.now())
    const onResize = () => { measure(); draw(performance.now()) }

    measure()
    draw(performance.now())
    addEventListener('scroll', onScroll, { passive: true })
    addEventListener('resize', onResize)
    const ro = new ResizeObserver(onResize)
    ro.observe(document.body)
    if (!reduced) raf = requestAnimationFrame(loop)
    return () => {
      cancelAnimationFrame(raf); ro.disconnect()
      removeEventListener('scroll', onScroll); removeEventListener('resize', onResize)
    }
  }, [reduced])

  return <canvas ref={ref} className="pixel-bg" aria-hidden="true" />
}
