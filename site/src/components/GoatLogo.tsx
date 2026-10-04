type Props = { className?: string; title?: string }

// Geometric goat head: swept-back horns, sideways ears, long face and beard.
export function GoatLogo({ className, title = 'goatdb goat' }: Props) {
  return (
    <svg className={className} viewBox="0 0 200 210" role="img" aria-label={title} xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="goat-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#56565d" />
          <stop offset="1" stopColor="#2a2a2f" />
        </linearGradient>
      </defs>
      <g fill="url(#goat-fill)" stroke="#6a6a73" strokeWidth="1.2" strokeLinejoin="round">
        {/* horns */}
        <path d="M84 64 C78 36 58 14 30 12 C48 22 60 40 66 70 Z" />
        <path d="M116 64 C122 36 142 14 170 12 C152 22 140 40 134 70 Z" />
        {/* ears */}
        <path d="M70 80 C54 72 34 74 18 86 C34 96 54 96 72 92 Z" />
        <path d="M130 80 C146 72 166 74 182 86 C166 96 146 96 128 92 Z" />
        {/* head */}
        <path d="M66 70 C70 58 130 58 134 70 L128 126 C126 148 114 162 100 164 C86 162 74 148 72 126 Z" />
        {/* beard */}
        <path d="M86 156 C90 176 94 192 100 206 C106 192 110 176 114 156 C108 162 92 162 86 156 Z" />
      </g>
      {/* eyes and nostrils */}
      <g fill="#0a0a0b">
        <ellipse cx="86" cy="98" rx="7.5" ry="3.6" transform="rotate(8 86 98)" />
        <ellipse cx="114" cy="98" rx="7.5" ry="3.6" transform="rotate(-8 114 98)" />
        <ellipse cx="92" cy="146" rx="3" ry="4.5" />
        <ellipse cx="108" cy="146" rx="3" ry="4.5" />
      </g>
    </svg>
  )
}
