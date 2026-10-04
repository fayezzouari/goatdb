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

  // Start muted when at least half the video is on screen (browsers only allow
  // muted autoplay) and pause when it scrolls away. Once the visitor pauses it
  // themselves, leave it alone.
  useEffect(() => {
    const video = ref.current!
    if (reduced) return
    let userPaused = false
    let autoPausing = false
    // the pause event fires asynchronously, so it consumes the flag itself
    const onPause = () => {
      if (autoPausing) { autoPausing = false; return }
      if (!video.ended) userPaused = true
    }
    const onPlay = () => { userPaused = false }
    video.addEventListener('pause', onPause)
    video.addEventListener('play', onPlay)
    const io = new IntersectionObserver(([entry]) => {
      if (userPaused) return
      if (entry.isIntersecting) video.play().catch(() => {})
      else if (!video.paused) { autoPausing = true; video.pause() }
    }, { threshold: 0.5 })
    io.observe(video)
    return () => {
      io.disconnect()
      video.removeEventListener('pause', onPause)
      video.removeEventListener('play', onPlay)
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
