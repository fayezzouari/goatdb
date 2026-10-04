import { useEffect, useRef } from 'react'
import { YOUTUBE_ID } from '../data.ts'
import { usePrefersReducedMotion } from '../hooks.ts'
import { Window } from './Window.tsx'

const BASE = import.meta.env.BASE_URL

// The MP4 lives on the site-assets branch and is copied into public/ at build
// time, so it is served with the site but never ships in releases.
export function DemoVideo() {
  const ref = useRef<HTMLVideoElement>(null)
  const reduced = usePrefersReducedMotion()

  // Start muted when a quarter of the video is on screen (browsers only allow
  // muted autoplay) and pause when it scrolls away. A pause only counts as the
  // visitor's when they just clicked, tapped or pressed a key on the video;
  // browsers also pause on their own (background tab, power saving, buffering).
  useEffect(() => {
    const video = ref.current!
    if (reduced) return
    // React sets the muted property but not the attribute, which Safari checks
    video.muted = true
    video.defaultMuted = true
    video.setAttribute('muted', '')

    let visible = false
    let userPaused = false
    let lastInteraction = -Infinity
    const interacted = () => { lastInteraction = performance.now() }
    const tryPlay = () => { if (visible && !userPaused && video.paused) video.play().catch(() => {}) }
    const onPause = () => {
      if (performance.now() - lastInteraction < 1000 && !video.ended) userPaused = true
    }
    const onPlay = () => { userPaused = false }

    video.addEventListener('pointerdown', interacted)
    video.addEventListener('keydown', interacted)
    video.addEventListener('pause', onPause)
    video.addEventListener('play', onPlay)
    video.addEventListener('canplay', tryPlay)
    const onVisibility = () => { if (!document.hidden) tryPlay() }
    document.addEventListener('visibilitychange', onVisibility)

    const io = new IntersectionObserver(([entry]) => {
      visible = entry.isIntersecting
      if (visible) tryPlay()
      else if (!video.paused) video.pause()
    }, { threshold: 0.25 })
    io.observe(video)

    return () => {
      io.disconnect()
      document.removeEventListener('visibilitychange', onVisibility)
      video.removeEventListener('pointerdown', interacted)
      video.removeEventListener('keydown', interacted)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('play', onPlay)
      video.removeEventListener('canplay', tryPlay)
    }
  }, [reduced])

  return (
    <section id="demo" className="wrap">
      <h2>See it run in two and a half minutes</h2>
      <p className="intro">
        Installing, creating a collection, inserting, searching, the web UI, surviving a restart, and using goatdb as a
        library. Also on <a href={`https://youtu.be/${YOUTUBE_ID}`}>YouTube</a>.
      </p>
      <Window title="goatdb demo" className="video">
        <video ref={ref} className="frame" controls muted playsInline preload="metadata" poster={`${BASE}demo-poster.jpg`}>
          <source src={`${BASE}goatdb-demo.mp4`} type="video/mp4" />
          Your browser can't play this video. <a href={`https://youtu.be/${YOUTUBE_ID}`}>Watch it on YouTube</a>.
        </video>
      </Window>
    </section>
  )
}
