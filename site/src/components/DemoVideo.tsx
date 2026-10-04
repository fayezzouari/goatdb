import { useState } from 'react'
import { YOUTUBE_ID } from '../data.ts'
import { Window } from './Window.tsx'

// YouTube loads only after the visitor presses play.
export function DemoVideo() {
  const [playing, setPlaying] = useState(false)
  return (
    <section id="demo" className="wrap">
      <h2>See it run in two and a half minutes</h2>
      <p className="intro">
        Installing, creating a collection, inserting, searching, the web UI, surviving a restart, and using goatdb as a
        library. Also on <a href={`https://youtu.be/${YOUTUBE_ID}`}>YouTube</a>.
      </p>
      <Window title="goatdb demo" className="video">
        <div className="frame">
          {playing ? (
            <iframe
              src={`https://www.youtube-nocookie.com/embed/${YOUTUBE_ID}?autoplay=1&rel=0&modestbranding=1`}
              title="goatdb demo"
              allow="autoplay; encrypted-media; picture-in-picture; fullscreen"
            />
          ) : (
            <img src={`${import.meta.env.BASE_URL}demo-poster.jpg`} alt="goatdb web UI showing search results for the movies collection" width={1280} height={720} />
          )}
        </div>
        {!playing && (
          <button className="playbtn" aria-label="Play the demo video (loads YouTube)" onClick={() => setPlaying(true)}>
            <span><svg viewBox="0 0 24 24"><path d="M6 4l15 8-15 8z" /></svg></span>
          </button>
        )}
      </Window>
    </section>
  )
}
