// Step 1: load a webpage. Always visible, so the current source is clear.

import { FormEvent, useState } from 'react'
import {
  BackendInfo,
  EXAMPLE_PAGE,
  IngestResult,
  fetcherLabel,
  hostname,
  postJson,
  requestErrorMessage,
} from './api'
import { ExampleChips, ThinkingDots, useElapsed } from './ui'

export function SourceBar({
  source,
  info,
  onIngested,
}: {
  source: IngestResult | null
  info: BackendInfo | null
  onIngested: (result: IngestResult) => void
}) {
  const [url, setUrl] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [isEditing, setIsEditing] = useState(false)
  const elapsed = useElapsed(isLoading)

  const ingest = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError(null)
    setIsLoading(true)
    try {
      const data = await postJson<IngestResult>(
        '/api/ingest',
        { url: url.trim() },
        'The website could not be ingested.',
      )
      onIngested(data)
      setIsEditing(false)
      setUrl('')
    } catch (caught) {
      setError(requestErrorMessage(caught, 'The website could not be ingested.'))
    } finally {
      setIsLoading(false)
    }
  }

  const showForm = !source || isEditing

  return (
    <section className="card source" aria-labelledby="source-heading">
      <div className="source__head">
        <span className={`step-badge${source ? ' step-badge--done' : ''}`} aria-hidden="true">
          {source ? '✓' : '1'}
        </span>
        <div>
          <h2 id="source-heading">{source ? 'Knowledge source' : 'Load a webpage'}</h2>
          <p className="muted">
            {source
              ? 'Answers and evaluations use only this page.'
              : `Paste a public URL. ${fetcherLabel(info)} extracts its main content, which is split into chunks and embedded for search.`}
          </p>
        </div>
      </div>

      {source && !isEditing && (
        <div className="source__loaded fade-in">
          <div className="source__title">
            <strong>{source.title || 'Untitled webpage'}</strong>
            <a href={source.url} target="_blank" rel="noreferrer">{hostname(source.url)} ↗</a>
          </div>
          <dl className="stats">
            <div><dt>Characters</dt><dd>{source.character_count.toLocaleString()}</dd></div>
            <div><dt>Chunks</dt><dd>{source.chunk_count.toLocaleString()}</dd></div>
          </dl>
          <button type="button" className="button button--ghost" onClick={() => setIsEditing(true)}>
            Change page
          </button>
        </div>
      )}

      {showForm && (
        <form className="source__form fade-in" onSubmit={ingest} aria-busy={isLoading}>
          <label className="visually-hidden" htmlFor="source-url">Webpage URL</label>
          <div className="input-row">
            <input
              id="source-url"
              type="url"
              value={url}
              onChange={(event) => setUrl(event.target.value)}
              placeholder="https://example.com/article"
              aria-invalid={Boolean(error)}
              disabled={isLoading}
              required
            />
            <button type="submit" className="button" disabled={isLoading || !url.trim()}>
              {isLoading ? 'Loading…' : 'Load page'}
            </button>
            {isEditing && (
              <button
                type="button"
                className="button button--ghost"
                onClick={() => { setIsEditing(false); setError(null) }}
                disabled={isLoading}
              >
                Cancel
              </button>
            )}
          </div>
          <ExampleChips
            label="Try"
            items={[{ text: EXAMPLE_PAGE.label, value: EXAMPLE_PAGE.url }]}
            onPick={setUrl}
            disabled={isLoading}
          />
          {isLoading && (
            <p className="progress-note">
              <ThinkingDots label={`Fetching, chunking and embedding the page… ${elapsed}s`} />
            </p>
          )}
          {error && <p className="alert alert--error" role="alert">{error}</p>}
        </form>
      )}
    </section>
  )
}
