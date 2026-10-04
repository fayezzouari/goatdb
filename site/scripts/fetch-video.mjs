// Downloads the demo video from the site-assets branch into public/ before a
// build. The MP4 is kept off master so it never ships in releases, so hosts
// that build from master (Vercel, CI) need to fetch it. Skips the download
// when the file is already present.
import { createWriteStream, existsSync, mkdirSync, renameSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { Readable } from 'node:stream'
import { pipeline } from 'node:stream/promises'
import { fileURLToPath } from 'node:url'

const URL = 'https://raw.githubusercontent.com/fayezzouari/goatdb/site-assets/goatdb-demo.mp4'
const out = join(dirname(fileURLToPath(import.meta.url)), '..', 'public', 'goatdb-demo.mp4')

if (existsSync(out) && statSync(out).size > 0) {
  console.log(`fetch-video: ${out} already present`)
  process.exit(0)
}

mkdirSync(dirname(out), { recursive: true })
const res = await fetch(URL)
if (!res.ok || !res.body) {
  console.error(`fetch-video: GET ${URL} failed with ${res.status}`)
  process.exit(1)
}
const tmp = `${out}.part`
await pipeline(Readable.fromWeb(res.body), createWriteStream(tmp))
renameSync(tmp, out)
console.log(`fetch-video: downloaded ${(statSync(out).size / 1e6).toFixed(1)} MB to ${out}`)
