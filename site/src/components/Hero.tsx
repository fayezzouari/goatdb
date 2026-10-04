import { INSTALL_CMD } from '../data.ts'
import { CopyButton } from './CopyButton.tsx'

export function Hero() {
  return (
    <>
      <h1 className="wordmark" aria-label="goatdb">
        {[...'goatdb'].map((ch, i) => (
          <span key={i} aria-hidden="true" style={{ animationDelay: `${i * 70 + 120}ms` }}>{ch}</span>
        ))}
      </h1>
      <p className="lede">A vector database you run as a server or import as a Go library.</p>
      <p className="sub">
        Store embeddings with metadata and find the closest ones in under a millisecond. One binary gives you a REST
        API, a web UI and durable storage. Built from scratch in Go.
      </p>
      <p className="by">By Fayez Zouari</p>
      <div className="actions">
        <div className="cmd">
          <span className="pr">$</span>
          <code>{INSTALL_CMD}</code>
          <CopyButton text={INSTALL_CMD} />
        </div>
      </div>
    </>
  )
}
