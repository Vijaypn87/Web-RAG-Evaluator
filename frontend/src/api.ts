// Types and HTTP helpers for talking to the FastAPI backend.

export const apiBaseUrl = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

export const REPOSITORY_URL = 'https://github.com/Vijaypn87/Web-RAG-Evaluator'

export type IngestResult = {
  url: string
  markdown_preview: string
  character_count: number
  chunk_count: number
  title: string | null
}

export type Citation = {
  label: string
  source_url: string
  chunk_number: number
}

export type RetrievedChunk = {
  label: string
  text: string
  source_url: string
  chunk_number: number
  distance: number
}

export type AskResult = {
  answer: string
  citations: Citation[]
  retrieved_chunks: RetrievedChunk[]
}

export type EvaluationCaseInput = {
  id: number
  question: string
  expected_answer: string
}

export type EvaluationResult = {
  question: string
  expected_answer: string
  generated_answer: string
  retrieval_relevance: number
  groundedness: number
  answer_correctness: number
  citation_validity: number
  overall_score: number
  passed: boolean
  explanations?: Record<string, string>
  error?: string | null
}

export type BackendInfo = {
  llm_provider: string
  chat_model: string
  embedding_model: string
  page_fetcher: string
}

export type BackendStatus = 'connecting' | 'waking' | 'ready' | 'offline'

export const MAX_EVALUATION_CASES = 10
export const PASS_THRESHOLD = 0.7

function apiDetail(data: unknown, fallback: string) {
  if (!data || typeof data !== 'object' || !('detail' in data)) return fallback

  const { detail } = data as { detail?: unknown }
  if (typeof detail === 'string') return detail
  if (!Array.isArray(detail)) return fallback

  const messages = detail.flatMap((item) => (
    item && typeof item === 'object' && 'msg' in item ? [String(item.msg)] : []
  ))
  return messages.join(' ') || fallback
}

export async function postJson<T>(path: string, body: unknown, fallback: string) {
  const response = await fetch(`${apiBaseUrl}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  const data: unknown = await response.json().catch(() => null)

  if (!response.ok) throw new Error(apiDetail(data, fallback))
  if (data === null) throw new Error(fallback)
  return data as T
}

export function requestErrorMessage(error: unknown, fallback: string) {
  if (error instanceof TypeError) {
    return 'Cannot reach the backend. Make sure the FastAPI server is running.'
  }
  return error instanceof Error ? error.message : fallback
}

// Free hosting puts the backend to sleep when idle; the first request wakes
// it, which can take up to a minute. Keep trying for a while before giving up.
export async function loadBackendInfo(signal: AbortSignal): Promise<BackendInfo> {
  const deadline = Date.now() + 120_000
  let lastError: unknown = null
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${apiBaseUrl}/api/info`, { signal })
      if (response.ok) return (await response.json()) as BackendInfo
      lastError = new Error(`HTTP ${response.status}`)
    } catch (error) {
      if (signal.aborted) throw error
      lastError = error
    }
    await new Promise((resolve) => setTimeout(resolve, 3_000))
  }
  throw lastError ?? new Error('Backend unavailable')
}

export function providerLabel(info: BackendInfo | null) {
  if (!info) return 'Language model'
  if (info.llm_provider === 'gemini') return 'Google Gemini'
  if (info.llm_provider === 'ollama') return 'Ollama'
  return 'OpenAI'
}

export function fetcherLabel(info: BackendInfo | null) {
  if (!info) return 'Page extractor'
  return info.page_fetcher === 'contextdev' ? 'Context.dev' : 'Trafilatura'
}

// Chroma returns cosine distance (0 = identical); show it as similarity.
export function similarity(distance: number) {
  return Math.max(0, Math.min(1, 1 - distance))
}

export function hostname(url: string) {
  try {
    return new URL(url).hostname.replace(/^www\./, '')
  } catch {
    return url
  }
}

// One-click examples, all about the same Wikipedia article.
export const EXAMPLE_PAGE = {
  label: 'Wikipedia · Retrieval-augmented generation',
  url: 'https://en.wikipedia.org/wiki/Retrieval-augmented_generation',
}

export const EXAMPLE_QUESTIONS = [
  'What is prompt stuffing?',
  'Does RAG prevent hallucinations?',
  'What trade-off does chunk size involve?',
]

export const EXAMPLE_CASES: Omit<EvaluationCaseInput, 'id'>[] = [
  {
    question: 'In what year was retrieval-augmented generation first proposed?',
    expected_answer: 'It was first proposed in 2020.',
  },
  {
    question: 'What is the more reproducible version of Retro called?',
    expected_answer: 'Retro++, which includes in-context RAG.',
  },
  {
    question: 'Where are the embeddings stored so that documents can be retrieved?',
    expected_answer: 'In a vector database.',
  },
]
