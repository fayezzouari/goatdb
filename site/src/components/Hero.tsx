import { INSTALL_CMD, REPO_URL } from '../data.ts'
import { CopyButton } from './CopyButton.tsx'
import { GitHubIcon } from './Nav.tsx'
import { Playground } from './Playground.tsx'

export function Hero() {
  return (
    <div className="wrap hero">
      <div className="hero-copy">
        <h1>Vector search, built from scratch in Go.</h1>
        <p className="sub">
          Store embeddings with metadata and find the nearest ones in under a millisecond. Run goatdb as a server with a
          REST API and web UI, or import it as a Go library.
        </p>
        <div className="ctas">
          <a className="btn btn-primary" href="#start">Get started</a>
          <a className="btn btn-ghost" href={REPO_URL}><GitHubIcon />View on GitHub</a>
        </div>
        <div className="cmd">
          <span className="pr">$</span>
          <code>{INSTALL_CMD}</code>
          <CopyButton text={INSTALL_CMD} />
        </div>
        <p className="by">By Fayez Zouari</p>
      </div>
      <div className="hero-demo">
        <Playground />
      </div>
    </div>
  )
}
