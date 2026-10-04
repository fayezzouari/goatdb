type Props = { className?: string; title?: string }

// Pixel-art goat in profile, facing left. One character per pixel:
//   h horn  b body  L back highlight  d shade  l ear/tail  e eye  n nose  k hoof  f far leg
const SPRITE = [
  '...........hhh..................',
  '..........hhh.h.................',
  '.........hhh....................',
  '........hhh.....................',
  '......bhhb......................',
  '.....bbbbbb.....................',
  '....bebbbbbll...................',
  '..bbbbbbbbbbll..................',
  '.nbbbbbbbbbb....................',
  '..bbbbbbbbbbb...................',
  '...dd..bbbbbb...................',
  '...dd..bbbbbbb..............ll..',
  '....d..bbbbbbbLLLLLLLLLLLLLLl...',
  '.......bbbbbbbbbbbbbbbbbbbbbb...',
  '.......bbbbbbbbbbbbbbbbbbbbbb...',
  '.......bbbbbbbbbbbbbbbbbbbbbb...',
  '........bbbbbbbbbbbbbbbbbbbbb...',
  '.........dddddddddddddddddddd...',
  '.........bb.ff.......bb.ff......',
  '.........bb.ff.......bb.ff......',
  '.........bb.ff.......bb.ff......',
  '.........kk.kk.......kk.kk......',
]

const COLORS: Record<string, string> = {
  h: '#8e8e97', b: '#5c5c64', L: '#6f6f78', d: '#3d3d44', l: '#77777f',
  e: '#0a0a0b', n: '#26262b', k: '#26262b', f: '#47474e',
}

export function GoatLogo({ className, title = 'goatdb goat' }: Props) {
  const w = SPRITE[0].length, h = SPRITE.length
  return (
    <svg className={className} viewBox={`0 0 ${w} ${h}`} role="img" aria-label={title}
         shapeRendering="crispEdges" xmlns="http://www.w3.org/2000/svg">
      {SPRITE.flatMap((row, y) =>
        [...row].map((c, x) => (c === '.' ? null : <rect key={`${x}-${y}`} x={x} y={y} width={1.02} height={1.02} fill={COLORS[c]} />)),
      )}
    </svg>
  )
}
