import { useEffect, useState } from 'react'

type Props = { text: string; className?: string }

export function CopyButton({ text, className = '' }: Props) {
  const [state, setState] = useState<'idle' | 'copied' | 'failed'>('idle')

  useEffect(() => {
    if (state === 'idle') return
    const t = setTimeout(() => setState('idle'), 1600)
    return () => clearTimeout(t)
  }, [state])

  async function copy() {
    try {
      await navigator.clipboard.writeText(text)
      setState('copied')
    } catch {
      setState('failed')
    }
  }

  const label = state === 'copied' ? 'Copied' : state === 'failed' ? 'Select and copy' : 'Copy'
  return (
    <button className={`copy ${state === 'copied' ? 'done' : ''} ${className}`} onClick={copy}>
      {label}
    </button>
  )
}
