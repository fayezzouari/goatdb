const INDEXES = [
  ['HNSW', 'Graph index. Best recall for the speed, the default choice.'],
  ['IVF', 'Clusters vectors with k-means and searches the closest lists.'],
  ['LSH', 'Random hyperplane hashing. Cheap to build.'],
  ['Flat', 'Exact brute-force search, for small sets and ground truth.'],
]
const METRICS = ['cosine', 'euclidean', 'dot_product', 'manhattan']
const WRITE_PATH = [
  ['Write-ahead log', 'Appended with a checksum and fsynced once per batch.'],
  ['Slot and metadata', 'Stored together in one BoltDB transaction.'],
  ['Memory-mapped file', 'Embeddings written to a flat mmapped vector file.'],
  ['Index', 'HNSW inserts run in parallel across your cores.'],
]

export function Features() {
  return (
    <section id="features" className="wrap">
      <h2>Everything a vector store needs, nothing it doesn't</h2>
      <div className="features">
        <div className="f">
          <h3>Four index types</h3>
          <p>Pick the trade-off per collection. Tune HNSW and IVF parameters at creation time and search depth per query.</p>
          <table className="idx">
            <tbody>
              {INDEXES.map(([name, desc]) => (
                <tr key={name}><td>{name}</td><td>{desc}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="f">
          <h3>Fast distance math</h3>
          <p>
            Hand-written AVX2 kernels on x86 and NEON on ARM, including Apple Silicon and Graviton. Cosine vectors are
            normalised once at insert, so each comparison is a single dot product.
          </p>
          <div className="metrics">{METRICS.map(m => <span key={m}>{m}</span>)}</div>
          <h3 style={{ marginTop: 40 }}>Three ways in</h3>
          <p>A JSON REST API, a built-in web UI at <code>/ui</code>, and the same engine as a Go package with no server at all.</p>
        </div>
        <div className="f wide">
          <h3>Durable writes</h3>
          <p>Every insert follows the same path. A crash at any point is recovered on the next start.</p>
          <ol className="path">
            {WRITE_PATH.map(([title, desc]) => (
              <li className="step" key={title}><b>{title}</b><span>{desc}</span></li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  )
}
