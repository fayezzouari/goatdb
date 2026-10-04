import { REPO_URL } from '../data.ts'
import { GoatLogo } from './GoatLogo.tsx'

export function Footer() {
  return (
    <footer className="wrap">
      <GoatLogo className="goat" />
      <div className="foot-row">
        <div><a className="brand" href="#">goatdb</a><br />Built by Fayez Zouari.</div>
        <div><a href={REPO_URL}>Source on GitHub</a><br /><a href={`${REPO_URL}/releases`}>Releases</a></div>
      </div>
    </footer>
  )
}
