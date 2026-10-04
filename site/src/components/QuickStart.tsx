import { useRef, useState, type KeyboardEvent, type ReactNode } from 'react'
import { CopyButton } from './CopyButton.tsx'
import { Window } from './Window.tsx'

type Tab = { id: string; label: string; title: string; copy: string; code: ReactNode }

const P = () => <span className="p">$ </span>

const TABS: Tab[] = [
  {
    id: 'docker', label: 'Docker', title: 'zsh',
    copy: `docker run -d -p 8080:8080 -v goatdb:/data ghcr.io/fayezzouari/goatdb
curl -X POST localhost:8080/collections \\
    -d '{"name": "docs", "dim": 768, "metric": "cosine", "index_type": "hnsw"}'`,
    code: (
      <>
        <P />docker run -d -p 8080:8080 -v goatdb:/data ghcr.io/fayezzouari/goatdb{'\n'}
        <P />curl -X POST localhost:8080/collections \{'\n'}
        {'    -d '}<span className="s">'{'{"name": "docs", "dim": 768, "metric": "cosine", "index_type": "hnsw"}'}'</span>{'\n'}
        <span className="c"># web UI: http://localhost:8080/ui</span>
      </>
    ),
  },
  {
    id: 'binary', label: 'Binary', title: 'zsh',
    copy: `go install github.com/fayezzouari/goatdb/cmd/goatdb@latest
goatdb -addr :8080 -dir ./data`,
    code: (
      <>
        <P />go install github.com/fayezzouari/goatdb/cmd/goatdb@latest{'\n'}
        <P />goatdb -addr :8080 -dir ./data{'\n'}
        <span className="c"># or download a prebuilt binary for Linux or macOS from GitHub Releases</span>
      </>
    ),
  },
  {
    id: 'lib', label: 'Go library', title: 'main.go',
    copy: `import (
    "github.com/fayezzouari/goatdb/core"
    "github.com/fayezzouari/goatdb/db"
)

store, _ := db.Open("./vectors")
defer store.Close()

docs, _ := store.CreateCollection("docs", 768, core.Cosine, "hnsw")
docs.AddVector(ctx, "doc-1", core.Vector{Embeddings: emb, Metadata: meta})
results, _ := docs.Search(ctx, core.Vector{Embeddings: query}, 10)`,
    code: (
      <>
        <span className="k">import</span> ({'\n'}
        {'    '}<span className="s">"github.com/fayezzouari/goatdb/core"</span>{'\n'}
        {'    '}<span className="s">"github.com/fayezzouari/goatdb/db"</span>{'\n'}
        ){'\n\n'}
        store, _ := db.<span className="t">Open</span>(<span className="s">"./vectors"</span>){'\n'}
        <span className="k">defer</span> store.<span className="t">Close</span>(){'\n\n'}
        docs, _ := store.<span className="t">CreateCollection</span>(<span className="s">"docs"</span>, <span className="n">768</span>, core.Cosine, <span className="s">"hnsw"</span>){'\n'}
        docs.<span className="t">AddVector</span>(ctx, <span className="s">"doc-1"</span>, core.Vector{'{'}Embeddings: emb, Metadata: meta{'}'}){'\n'}
        results, _ := docs.<span className="t">Search</span>(ctx, core.Vector{'{'}Embeddings: query{'}'}, <span className="n">10</span>)
      </>
    ),
  },
]

export function QuickStart() {
  const [active, setActive] = useState(0)
  const refs = useRef<(HTMLButtonElement | null)[]>([])

  function onKey(e: KeyboardEvent, i: number) {
    const d = e.key === 'ArrowRight' ? 1 : e.key === 'ArrowLeft' ? -1 : 0
    if (!d) return
    const n = (i + d + TABS.length) % TABS.length
    setActive(n)
    refs.current[n]?.focus()
  }

  return (
    <section id="start" className="wrap">
      <h2>Get started</h2>
      <div className="tabs">
        <div className="tablist" role="tablist" aria-label="Install options">
          {TABS.map((t, i) => (
            <button
              key={t.id} ref={el => { refs.current[i] = el }} role="tab" id={`t-${t.id}`}
              aria-selected={i === active} aria-controls={`p-${t.id}`} tabIndex={i === active ? 0 : -1}
              onClick={() => setActive(i)} onKeyDown={e => onKey(e, i)}
            >
              {t.label}
            </button>
          ))}
        </div>
        {TABS.map((t, i) => (
          <div key={t.id} className="panel" role="tabpanel" id={`p-${t.id}`} aria-labelledby={`t-${t.id}`} hidden={i !== active}>
            <Window title={t.title}>
              <CopyButton text={t.copy} />
              <pre>{t.code}</pre>
            </Window>
          </div>
        ))}
      </div>
    </section>
  )
}
