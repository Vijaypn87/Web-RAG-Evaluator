// Small shared UI pieces: timers, loaders, score bars and answer rendering.

import { Fragment, ReactNode, useEffect, useRef, useState } from 'react'

const prefersReducedMotion = () =>
  typeof window !== 'undefined'
  && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches

/** Seconds elapsed while `running` is true (resets when it starts again). */
export function useElapsed(running: boolean) {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    if (!running) return
    setSeconds(0)
    const started = Date.now()
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 250)
    return () => clearInterval(timer)
  }, [running])
  return seconds
}

/** Animate a number from 0 to `value` once it appears. */
export function CountUp({ value, decimals = 2 }: { value: number; decimals?: number }) {
  const [shown, setShown] = useState(prefersReducedMotion() ? value : 0)
  const frame = useRef<number | null>(null)

  useEffect(() => {
    if (prefersReducedMotion()) {
      setShown(value)
      return
    }
    const duration = 700
    const start = performance.now()
    const step = (now: number) => {
      const progress = Math.min(1, (now - start) / duration)
      const eased = 1 - (1 - progress) ** 3
      setShown(value * eased)
      if (progress < 1) frame.current = requestAnimationFrame(step)
    }
    frame.current = requestAnimationFrame(step)
    return () => {
      if (frame.current) cancelAnimationFrame(frame.current)
    }
  }, [value])

  return <>{shown.toFixed(decimals)}</>
}

export function ThinkingDots({ label }: { label: string }) {
  return (
    <span className="thinking" role="status">
      <span className="thinking__dots" aria-hidden="true"><i /><i /><i /></span>
      {label}
    </span>
  )
}

export function Spinner() {
  return <span className="spinner" aria-hidden="true" />
}

function scoreTone(score: number) {
  if (score >= 0.8) return 'good'
  if (score >= 0.5) return 'fair'
  return 'poor'
}

/** A horizontal bar that fills to `value` (0–1) with a short animation. */
export function ScoreBar({ value, label }: { value: number; label?: string }) {
  const percent = Math.round(Math.max(0, Math.min(1, value)) * 100)
  return (
    <span className={`score-bar score-bar--${scoreTone(value)}`} title={label}>
      <span className="score-bar__track">
        <span className="score-bar__fill" style={{ ['--fill' as string]: `${percent}%` }} />
      </span>
      <span className="score-bar__value">{value.toFixed(2)}</span>
    </span>
  )
}

/** Circular score gauge for the overall evaluation result. */
export function ScoreRing({ value }: { value: number }) {
  const radius = 34
  const circumference = 2 * Math.PI * radius
  const offset = circumference * (1 - Math.max(0, Math.min(1, value)))
  return (
    <div className={`score-ring score-ring--${scoreTone(value)}`}>
      <svg viewBox="0 0 80 80" aria-hidden="true">
        <circle className="score-ring__track" cx="40" cy="40" r={radius} />
        <circle
          className="score-ring__value"
          cx="40"
          cy="40"
          r={radius}
          strokeDasharray={circumference}
          style={{ ['--offset' as string]: offset, ['--full' as string]: circumference }}
        />
      </svg>
      <strong><CountUp value={value} /></strong>
    </div>
  )
}

/**
 * Render an answer: [S1]-style labels become clickable citation buttons and
 * simple **bold** / *italic* Markdown is shown as formatting, not asterisks.
 */
export function AnswerText({
  text,
  validLabels,
  activeLabel,
  onCite,
}: {
  text: string
  validLabels: Set<string>
  activeLabel: string | null
  onCite: (label: string) => void
}) {
  const parts = text.split(/(\[S\d+\]|\*\*[^*\n]+\*\*|\*[^*\n]+\*)/g)
  const nodes: ReactNode[] = parts.map((part, index) => {
    if (/^\[S\d+\]$/.test(part)) {
      if (!validLabels.has(part)) return null
      return (
        <button
          key={index}
          type="button"
          className={`cite${activeLabel === part ? ' cite--active' : ''}`}
          onClick={() => onCite(part)}
          aria-label={`Show source ${part}`}
        >
          {part.slice(1, -1)}
        </button>
      )
    }
    if (/^\*\*[^*]+\*\*$/.test(part)) return <strong key={index}>{part.slice(2, -2)}</strong>
    if (/^\*[^*]+\*$/.test(part)) return <em key={index}>{part.slice(1, -1)}</em>
    return <Fragment key={index}>{part}</Fragment>
  })
  return <p className="answer__text">{nodes}</p>
}

export function ExampleChips({
  label,
  items,
  onPick,
  disabled,
}: {
  label: string
  items: { text: string; value: string }[]
  onPick: (value: string) => void
  disabled?: boolean
}) {
  return (
    <div className="chips">
      <span className="chips__label">{label}</span>
      {items.map((item) => (
        <button
          key={item.value}
          type="button"
          className="chip"
          onClick={() => onPick(item.value)}
          disabled={disabled}
        >
          {item.text}
        </button>
      ))}
    </div>
  )
}
