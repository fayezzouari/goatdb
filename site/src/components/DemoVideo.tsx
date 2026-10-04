import { YOUTUBE_ID } from '../data.ts'
import { Window } from './Window.tsx'

const BASE = import.meta.env.BASE_URL

// The MP4 lives on the site-assets branch and is copied into public/ at build
// time, so it is served with the site but never ships in releases.
export function DemoVideo() {
  return (
    <section id="demo" className="wrap">
      <h2>See it run in two and a half minutes</h2>
      <p className="intro">
        Installing, creating a collection, inserting, searching, the web UI, surviving a restart, and using goatdb as a
        library. Also on <a href={`https://youtu.be/${YOUTUBE_ID}`}>YouTube</a>.
      </p>
      <Window title="goatdb demo" className="video">
        <video className="frame" controls preload="metadata" playsInline poster={`${BASE}demo-poster.jpg`}>
          <source src={`${BASE}goatdb-demo.mp4`} type="video/mp4" />
          Your browser can't play this video. <a href={`https://youtu.be/${YOUTUBE_ID}`}>Watch it on YouTube</a>.
        </video>
      </Window>
    </section>
  )
}
