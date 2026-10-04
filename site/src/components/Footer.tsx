import { REPO_URL } from '../data.ts'

export function Footer() {
  return (
    <footer className="wrap">
      <div><a className="brand" href="#">goatdb</a><br />Built by Fayez Zouari.</div>
      <div><a href={REPO_URL}>Source on GitHub</a><br /><a href={`${REPO_URL}/releases`}>Releases</a></div>
      <div>
        Demo soundtrack: “One Cool Minute” by Loyalty Freak Music,{' '}
        <a href="https://creativecommons.org/publicdomain/zero/1.0/">CC0</a>.
      </div>
    </footer>
  )
}
