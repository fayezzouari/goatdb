import type { ReactNode } from 'react'

type Props = { title: string; className?: string; children: ReactNode }

// Dark rounded window with macOS-style title bar, shared by every panel on the page.
export function Window({ title, className = '', children }: Props) {
  return (
    <div className={`window ${className}`}>
      <div className="bar">
        <i />
        <i />
        <i />
        <b>{title}</b>
      </div>
      {children}
    </div>
  )
}
