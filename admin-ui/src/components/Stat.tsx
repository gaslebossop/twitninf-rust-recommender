import type { ReactNode } from 'react'

export function Stat({
  label,
  value,
  unit,
  tone = 'default',
  hint,
}: {
  label: string
  value: ReactNode
  unit?: string
  tone?: 'default' | 'positive' | 'negative' | 'idle'
  hint?: string
}) {
  return (
    <div className="stat">
      <span className="stat-label">{label}</span>
      <span className={`stat-value mono tone-${tone}`}>
        {value}
        {unit && <span className="stat-unit">{unit}</span>}
      </span>
      {hint && <span className="stat-hint">{hint}</span>}
    </div>
  )
}

export function Panel({ title, action, children }: { title: string; action?: ReactNode; children: ReactNode }) {
  return (
    <section className="panel">
      <header className="panel-head">
        <h2>{title}</h2>
        {action}
      </header>
      <div className="panel-body">{children}</div>
    </section>
  )
}

export function Pill({ tone = 'idle', children }: { tone?: 'positive' | 'negative' | 'idle' | 'accent'; children: ReactNode }) {
  return <span className={`pill pill-${tone}`}>{children}</span>
}
