import { useEffect, useState } from 'react'
import {
  BackendInfo,
  BackendStatus,
  EvaluationCaseInput,
  EvaluationResult,
  IngestResult,
  REPOSITORY_URL,
  fetcherLabel,
  loadBackendInfo,
  providerLabel,
} from './api'
import { AskPanel, Turn } from './AskPanel'
import { EvalPanel, newCase } from './EvalPanel'
import { SourceBar } from './SourceBar'

type Mode = 'ask' | 'evaluate'

const STATUS_TEXT: Record<BackendStatus, string> = {
  connecting: 'Connecting',
  waking: 'Waking up',
  ready: 'Online',
  offline: 'Offline',
}

function App() {
  const [info, setInfo] = useState<BackendInfo | null>(null)
  const [status, setStatus] = useState<BackendStatus>('connecting')
  const [source, setSource] = useState<IngestResult | null>(null)
  const [mode, setMode] = useState<Mode>('ask')

  // Kept here so nothing is lost when switching between Ask and Evaluate.
  const [turns, setTurns] = useState<Turn[]>([])
  const [cases, setCases] = useState<EvaluationCaseInput[]>(() => [newCase(), newCase(), newCase()])
  const [results, setResults] = useState<EvaluationResult[] | null>(null)

  useEffect(() => {
    const controller = new AbortController()
    // If the backend has not answered after 2 seconds, it is probably asleep.
    const wakeTimer = setTimeout(() => {
      setStatus((current) => (current === 'connecting' ? 'waking' : current))
    }, 2_000)

    loadBackendInfo(controller.signal)
      .then((data) => {
        setInfo(data)
        setStatus('ready')
      })
      .catch(() => {
        if (!controller.signal.aborted) setStatus('offline')
      })
      .finally(() => clearTimeout(wakeTimer))

    return () => {
      controller.abort()
      clearTimeout(wakeTimer)
    }
  }, [])

  const onIngested = (result: IngestResult) => {
    setSource(result)
    // Old answers and scores belong to the previous page.
    setTurns(() => [])
    setResults(null)
  }

  return (
    <div className="app">
      <header className="topbar">
        <a className="brand" href="/" aria-label="Web RAG Evaluator home">
          <span className="brand__mark" aria-hidden="true"><i /><i /><i /></span>
          <span className="brand__name">Web RAG Evaluator</span>
        </a>
        <div className="topbar__right">
          <span className={`status status--${status}`} role="status">
            <span className="status__dot" aria-hidden="true" />
            {STATUS_TEXT[status]}
          </span>
          <a className="button button--ghost button--small" href={REPOSITORY_URL} target="_blank" rel="noreferrer">
            GitHub ↗
          </a>
        </div>
      </header>

      <main className="main">
        <section className="hero fade-in">
          <h1>Ask a webpage. <span className="gradient-text">Check the answer.</span></h1>
          <p className="muted">
            A small, inspectable retrieval-augmented generation pipeline: load a page,
            ask grounded questions with clickable citations, then score the system
            with an automatic evaluator.
          </p>
        </section>

        {status === 'waking' && (
          <p className="alert alert--info fade-in" role="status">
            Waking up the server… This free demo sleeps when nobody is using it,
            so the first visit can take up to a minute.
          </p>
        )}
        {status === 'offline' && (
          <p className="alert alert--error" role="alert">
            The backend is not responding right now. Please refresh in a minute.
          </p>
        )}

        <SourceBar source={source} info={info} onIngested={onIngested} />

        <section className={`card workspace${source ? '' : ' workspace--locked'}`} aria-label="Ask and evaluate">
          <div className="workspace__head">
            <span className={`step-badge${source ? '' : ' step-badge--locked'}`} aria-hidden="true">2</span>
            <div className="segmented" role="tablist" aria-label="Mode" data-active={mode}>
              <span className="segmented__indicator" aria-hidden="true" />
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'ask'}
                className={mode === 'ask' ? 'is-active' : ''}
                onClick={() => setMode('ask')}
              >
                Ask questions
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={mode === 'evaluate'}
                className={mode === 'evaluate' ? 'is-active' : ''}
                onClick={() => setMode('evaluate')}
              >
                Evaluate
                {results && <span className="badge">{results.filter((result) => result.passed).length}/{results.length}</span>}
              </button>
            </div>
          </div>

          {!source ? (
            <div className="locked fade-in">
              <p>Load a webpage above to start asking questions and running evaluations.</p>
            </div>
          ) : (
            <div key={mode} className="fade-in" role="tabpanel">
              {mode === 'ask' ? (
                <AskPanel turns={turns} setTurns={setTurns} info={info} />
              ) : (
                <EvalPanel cases={cases} setCases={setCases} results={results} setResults={setResults} />
              )}
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        <span>FastAPI · React · Chroma · {providerLabel(info)} · {fetcherLabel(info)}</span>
        <a href={REPOSITORY_URL} target="_blank" rel="noreferrer">Source on GitHub</a>
      </footer>
    </div>
  )
}

export default App
