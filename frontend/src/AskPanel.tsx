// Step 2: ask questions. Each answer keeps its sources beside it, and
// clicking a citation highlights the passage it came from.

import { FormEvent, KeyboardEvent, useRef, useState } from 'react'
import {
  AskResult,
  BackendInfo,
  EXAMPLE_QUESTIONS,
  RetrievedChunk,
  postJson,
  providerLabel,
  requestErrorMessage,
  similarity,
} from './api'
import { AnswerText, ExampleChips, ThinkingDots, useElapsed } from './ui'

export type Turn = {
  id: number
  question: string
  status: 'loading' | 'done' | 'error'
  result?: AskResult
  error?: string
}

let nextTurnId = 1

export function AskPanel({
  turns,
  setTurns,
  info,
}: {
  turns: Turn[]
  setTurns: (update: (turns: Turn[]) => Turn[]) => void
  info: BackendInfo | null
}) {
  const [question, setQuestion] = useState('')
  const isLoading = turns.some((turn) => turn.status === 'loading')
  const formRef = useRef<HTMLFormElement>(null)

  const ask = async (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault()
    const text = question.trim()
    if (!text || isLoading) return

    const id = nextTurnId++
    setTurns((current) => [{ id, question: text, status: 'loading' }, ...current])
    setQuestion('')

    try {
      const result = await postJson<AskResult>(
        '/api/ask',
        { question: text },
        'The question could not be answered.',
      )
      setTurns((current) => current.map((turn) => (
        turn.id === id ? { ...turn, status: 'done', result } : turn
      )))
    } catch (caught) {
      const message = requestErrorMessage(caught, 'The question could not be answered.')
      setTurns((current) => current.map((turn) => (
        turn.id === id ? { ...turn, status: 'error', error: message } : turn
      )))
    }
  }

  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
      event.preventDefault()
      formRef.current?.requestSubmit()
    }
  }

  return (
    <div className="panel-body">
      <form ref={formRef} className="ask-form" onSubmit={ask}>
        <label className="visually-hidden" htmlFor="question">Question</label>
        <textarea
          id="question"
          rows={2}
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Ask something the page answers…"
          disabled={isLoading}
        />
        <div className="ask-form__bar">
          <ExampleChips
            label="Examples"
            items={EXAMPLE_QUESTIONS.map((text) => ({ text, value: text }))}
            onPick={setQuestion}
            disabled={isLoading}
          />
          <div className="ask-form__actions">
            <kbd className="hint">Ctrl + Enter</kbd>
            <button type="submit" className="button" disabled={isLoading || !question.trim()}>
              Ask
            </button>
          </div>
        </div>
      </form>

      {turns.length === 0 ? (
        <div className="empty fade-in">
          <p><strong>No questions yet.</strong> Try an example above. Answers cite the passages they use, and you can click a citation to see its source.</p>
        </div>
      ) : (
        <div className="thread">
          {turns.length > 1 && (
            <div className="thread__head">
              <span className="muted">{turns.length} questions</span>
              <button
                type="button"
                className="link-button"
                onClick={() => setTurns(() => [])}
                disabled={isLoading}
              >
                Clear
              </button>
            </div>
          )}
          {turns.map((turn) => (
            <TurnCard key={turn.id} turn={turn} info={info} />
          ))}
        </div>
      )}
    </div>
  )
}

function TurnCard({ turn, info }: { turn: Turn; info: BackendInfo | null }) {
  const [activeLabel, setActiveLabel] = useState<string | null>(null)
  const sourceRefs = useRef<Record<string, HTMLElement | null>>({})
  const elapsed = useElapsed(turn.status === 'loading')
  const [copied, setCopied] = useState(false)

  const cite = (label: string) => {
    setActiveLabel(label)
    sourceRefs.current[label]?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
  }

  const copy = async () => {
    if (!turn.result) return
    try {
      await navigator.clipboard.writeText(turn.result.answer)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // Clipboard can be blocked; nothing else to do.
    }
  }

  const result = turn.result
  const validLabels = new Set(result?.citations.map((citation) => citation.label) ?? [])

  return (
    <article className="turn slide-up">
      <p className="turn__question"><span aria-hidden="true">Q</span>{turn.question}</p>

      {turn.status === 'loading' && (
        <div className="turn__loading">
          <ThinkingDots label={`Retrieving passages and writing an answer… ${elapsed}s`} />
          <div className="skeleton" aria-hidden="true"><i /><i /><i /></div>
        </div>
      )}

      {turn.status === 'error' && (
        <p className="alert alert--error" role="alert">{turn.error}</p>
      )}

      {result && (
        <div className="turn__body">
          <div className="answer">
            <AnswerText
              text={result.answer}
              validLabels={validLabels}
              activeLabel={activeLabel}
              onCite={cite}
            />
            <div className="answer__meta">
              <span className="muted">
                {providerLabel(info)}{info ? ` · ${info.chat_model}` : ''}
              </span>
              <span className="muted">
                {result.citations.length
                  ? `${result.citations.length} citation${result.citations.length === 1 ? '' : 's'}`
                  : 'No citations'}
              </span>
              <button type="button" className="link-button" onClick={copy}>
                {copied ? 'Copied ✓' : 'Copy'}
              </button>
            </div>
          </div>

          <div className="sources" aria-label="Retrieved passages">
            <p className="sources__title">Retrieved passages</p>
            {result.retrieved_chunks.map((chunk) => (
              <SourceCard
                key={chunk.label}
                chunk={chunk}
                cited={validLabels.has(chunk.label)}
                active={activeLabel === chunk.label}
                onSelect={() => setActiveLabel(chunk.label === activeLabel ? null : chunk.label)}
                refCallback={(element) => { sourceRefs.current[chunk.label] = element }}
              />
            ))}
          </div>
        </div>
      )}
    </article>
  )
}

function SourceCard({
  chunk,
  cited,
  active,
  onSelect,
  refCallback,
}: {
  chunk: RetrievedChunk
  cited: boolean
  active: boolean
  onSelect: () => void
  refCallback: (element: HTMLElement | null) => void
}) {
  const [expanded, setExpanded] = useState(false)
  const score = similarity(chunk.distance)
  // Show Markdown links and emphasis as plain readable text.
  const plain = chunk.text
    .replace(/!?\[([^\]]*)\]\([^)]*\)/g, '$1')
    // A chunk can start or end in the middle of a link; drop the leftovers.
    .replace(/\]\([^)]*\)?/g, '')
    .replace(/\(https?:\/\/[^)\s]*\)?/g, '')
    .replace(/https?:\/\/\S+/g, '')
    .replace(/(\*\*|__)(.+?)\1/g, '$2')
    .replace(/^#{1,6}\s+/gm, '')
  const long = plain.length > 220
  const text = expanded || !long ? plain : `${plain.slice(0, 220)}…`

  return (
    <article
      ref={refCallback}
      className={`source-card${active ? ' source-card--active' : ''}${cited ? '' : ' source-card--uncited'}`}
    >
      <button type="button" className="source-card__head" onClick={onSelect} aria-pressed={active}>
        <span className="cite cite--static">{chunk.label.slice(1, -1)}</span>
        <span className="source-card__chunk">Chunk {chunk.chunk_number}</span>
        {!cited && <span className="tag">not cited</span>}
        <span className="relevance" title={`Cosine distance ${chunk.distance.toFixed(4)}`}>
          <span className="relevance__track">
            <span className="relevance__fill" style={{ ['--fill' as string]: `${Math.round(score * 100)}%` }} />
          </span>
          {Math.round(score * 100)}%
        </span>
      </button>
      <p className="source-card__text">{text}</p>
      {long && (
        <button type="button" className="link-button" onClick={() => setExpanded(!expanded)}>
          {expanded ? 'Show less' : 'Show more'}
        </button>
      )}
    </article>
  )
}
