// Step 3: evaluate. Write test cases, run them, and inspect every score.

import { FormEvent, Fragment, useState } from 'react'
import {
  EXAMPLE_CASES,
  EvaluationCaseInput,
  EvaluationResult,
  MAX_EVALUATION_CASES,
  PASS_THRESHOLD,
  postJson,
  requestErrorMessage,
} from './api'
import { ScoreBar, ScoreRing, ThinkingDots, useElapsed } from './ui'

const METRICS = [
  { key: 'retrieval_relevance', label: 'Retrieval', help: 'Are the retrieved passages relevant to the question?' },
  { key: 'groundedness', label: 'Grounded', help: 'Is the answer supported by the retrieved passages?' },
  { key: 'answer_correctness', label: 'Correct', help: 'Does the answer match the expected answer?' },
  { key: 'citation_validity', label: 'Citations', help: 'Share of written citations that point to a real passage.' },
] as const

type MetricKey = (typeof METRICS)[number]['key']

let nextCaseId = 100

export function newCase(question = '', expected_answer = ''): EvaluationCaseInput {
  return { id: nextCaseId++, question, expected_answer }
}

export function EvalPanel({
  cases,
  setCases,
  results,
  setResults,
}: {
  cases: EvaluationCaseInput[]
  setCases: (update: (cases: EvaluationCaseInput[]) => EvaluationCaseInput[]) => void
  results: EvaluationResult[] | null
  setResults: (results: EvaluationResult[] | null) => void
}) {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [openRow, setOpenRow] = useState<number | null>(null)
  const elapsed = useElapsed(isLoading)

  const update = (id: number, field: 'question' | 'expected_answer', value: string) => {
    setCases((current) => current.map((item) => (item.id === id ? { ...item, [field]: value } : item)))
  }

  const filled = cases.filter((item) => item.question.trim() && item.expected_answer.trim())
  const canRun = filled.length > 0 && filled.length === cases.length

  const run = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setError(null)
    setIsLoading(true)
    setOpenRow(null)
    try {
      const data = await postJson<EvaluationResult[]>(
        '/api/evaluate',
        cases.map((item) => ({
          question: item.question.trim(),
          expected_answer: item.expected_answer.trim(),
        })),
        'The evaluations could not be completed.',
      )
      setResults(data)
    } catch (caught) {
      setError(requestErrorMessage(caught, 'The evaluations could not be completed.'))
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="panel-body">
      <form className="cases" onSubmit={run} aria-busy={isLoading}>
        <div className="cases__toolbar">
          <p className="muted">
            Each case runs retrieval, answering and an LLM judge. A case passes at an
            overall score of {PASS_THRESHOLD.toFixed(2)} or more.
          </p>
          <button
            type="button"
            className="button button--ghost"
            onClick={() => {
              setCases(() => EXAMPLE_CASES.map((item) => newCase(item.question, item.expected_answer)))
              setResults(null)
            }}
            disabled={isLoading}
          >
            Load examples
          </button>
        </div>

        <ol className="case-list">
          {cases.map((item, index) => (
            <li key={item.id} className="case slide-up">
              <span className="case__number">{index + 1}</span>
              <label>
                <span className="field-label">Question</span>
                <textarea
                  rows={2}
                  value={item.question}
                  onChange={(event) => update(item.id, 'question', event.target.value)}
                  placeholder="What should the page answer?"
                  disabled={isLoading}
                />
              </label>
              <label>
                <span className="field-label">Expected answer</span>
                <textarea
                  rows={2}
                  value={item.expected_answer}
                  onChange={(event) => update(item.id, 'expected_answer', event.target.value)}
                  placeholder="The correct answer, in your words"
                  disabled={isLoading}
                />
              </label>
              <button
                type="button"
                className="icon-button"
                onClick={() => setCases((current) => current.filter((other) => other.id !== item.id))}
                disabled={isLoading || cases.length === 1}
                aria-label={`Remove case ${index + 1}`}
                title="Remove case"
              >
                ×
              </button>
            </li>
          ))}
        </ol>

        <div className="cases__actions">
          <button
            type="button"
            className="link-button"
            onClick={() => setCases((current) => [...current, newCase()])}
            disabled={isLoading || cases.length >= MAX_EVALUATION_CASES}
          >
            + Add case ({cases.length}/{MAX_EVALUATION_CASES})
          </button>
          <button type="submit" className="button" disabled={isLoading || !canRun}>
            {isLoading ? 'Running…' : `Run ${cases.length} evaluation${cases.length === 1 ? '' : 's'}`}
          </button>
        </div>
        {isLoading && (
          <p className="progress-note">
            <ThinkingDots label={`Answering and judging ${cases.length} case${cases.length === 1 ? '' : 's'}… ${elapsed}s`} />
          </p>
        )}
        {error && <p className="alert alert--error" role="alert">{error}</p>}
      </form>

      {results && <Results results={results} openRow={openRow} setOpenRow={setOpenRow} />}
    </div>
  )
}

function average(values: number[]) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0
}

function Results({
  results,
  openRow,
  setOpenRow,
}: {
  results: EvaluationResult[]
  openRow: number | null
  setOpenRow: (row: number | null) => void
}) {
  const scored = results.filter((result) => !result.error)
  const passed = scored.filter((result) => result.passed).length
  const failedRuns = results.length - scored.length

  return (
    <section className="results fade-in" aria-label="Evaluation results">
      <div className="summary">
        <ScoreRing value={average(scored.map((result) => result.overall_score))} />
        <div className="summary__text">
          <p className="summary__headline">
            <strong>{passed}/{results.length}</strong> passed
          </p>
          <p className="muted">
            Average overall score
            {failedRuns > 0 && ` · ${failedRuns} case${failedRuns === 1 ? '' : 's'} could not run`}
          </p>
        </div>
        <dl className="summary__metrics">
          {METRICS.map((metric) => (
            <div key={metric.key} title={metric.help}>
              <dt>{metric.label}</dt>
              <dd>
                <ScoreBar value={average(scored.map((result) => result[metric.key as MetricKey]))} />
              </dd>
            </div>
          ))}
        </dl>
      </div>

      <div className="table-wrap">
        <table className="results-table">
          <thead>
            <tr>
              <th scope="col">Case</th>
              {METRICS.map((metric) => <th key={metric.key} scope="col" title={metric.help}>{metric.label}</th>)}
              <th scope="col">Overall</th>
              <th scope="col"><span className="visually-hidden">Details</span></th>
            </tr>
          </thead>
          <tbody>
            {results.map((result, index) => {
              const open = openRow === index
              return (
                <Fragment key={`${result.question}-${index}`}>
                  <tr
                    className={`row${open ? ' row--open' : ''}`}
                    onClick={() => setOpenRow(open ? null : index)}
                  >
                    <td className="row__question">
                      <span className={`dot dot--${result.error ? 'error' : result.passed ? 'pass' : 'fail'}`} aria-hidden="true" />
                      {result.question}
                    </td>
                    {result.error ? (
                      <td colSpan={METRICS.length + 1} className="row__error">Could not run</td>
                    ) : (
                      <>
                        {METRICS.map((metric) => (
                          <td key={metric.key}><ScoreBar value={result[metric.key as MetricKey]} /></td>
                        ))}
                        <td>
                          <span className={`pill pill--${result.passed ? 'pass' : 'fail'}`}>
                            {result.overall_score.toFixed(2)} {result.passed ? 'Pass' : 'Fail'}
                          </span>
                        </td>
                      </>
                    )}
                    <td>
                      <button
                        type="button"
                        className="icon-button"
                        aria-expanded={open}
                        aria-label={open ? 'Hide details' : 'Show details'}
                        onClick={(event) => { event.stopPropagation(); setOpenRow(open ? null : index) }}
                      >
                        <span className={`chevron${open ? ' chevron--open' : ''}`} aria-hidden="true">›</span>
                      </button>
                    </td>
                  </tr>
                  {open && (
                    <tr className="details-row">
                      <td colSpan={METRICS.length + 3}>
                        <CaseDetails result={result} />
                      </td>
                    </tr>
                  )}
                </Fragment>
              )
            })}
          </tbody>
        </table>
      </div>
      <p className="muted small">Click a row to compare the answers and read the judge's reasons.</p>
    </section>
  )
}

function CaseDetails({ result }: { result: EvaluationResult }) {
  if (result.error) {
    return <p className="alert alert--error">{result.error}</p>
  }
  const explanations = result.explanations ?? {}
  return (
    <div className="details slide-down">
      <div className="details__answers">
        <div>
          <p className="field-label">Expected answer</p>
          <p>{result.expected_answer}</p>
        </div>
        <div>
          <p className="field-label">Generated answer</p>
          <p>{result.generated_answer}</p>
        </div>
      </div>
      <ul className="details__reasons">
        {METRICS.filter((metric) => explanations[metric.key]).map((metric) => (
          <li key={metric.key}>
            <span className="field-label">{metric.label}</span>
            {explanations[metric.key]}
          </li>
        ))}
        <li>
          <span className="field-label">Citations</span>
          Scored automatically: the share of [S#] labels that point to a supplied passage.
        </li>
      </ul>
    </div>
  )
}
