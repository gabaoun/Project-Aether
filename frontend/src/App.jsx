import { useState } from 'react'
import SearchForm from './components/SearchForm.jsx'
import ExampleQuestions from './components/ExampleQuestions.jsx'
import AnswerCard from './components/AnswerCard.jsx'
import { useAetherQuery } from './hooks/useAetherQuery.js'

export default function App() {
  const [query, setQuery] = useState('')
  const { ask, answer, sources, fromCache, loading, error } = useAetherQuery()

  function handleAsk(text) {
    const trimmed = text.trim()
    if (!trimmed) return
    setQuery(trimmed)
    ask(trimmed)
  }

  return (
    <main>
      <header>
        <div className="header-top">
          <div className="logo-badge">
            <div className="logo-icon">
              <svg viewBox="0 0 24 24">
                <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" />
              </svg>
            </div>
            Project Aether
          </div>
          <div className="nav-links">
            <a
              className="nav-link"
              href="https://github.com/gabaoun/Project-Aether"
              target="_blank"
              rel="noopener noreferrer"
            >
              GitHub
            </a>
            <a className="nav-link" href="/docs" target="_blank" rel="noreferrer">
              API Docs
            </a>
          </div>
        </div>
        <p>
          Event-driven RAG engine interactive live demo. Ask questions about
          Gabriel Penha's background, architecture decisions, or tech stack.
        </p>
      </header>

      <div className="card">
        <SearchForm
          query={query}
          onQueryChange={setQuery}
          onSubmit={handleAsk}
          loading={loading}
        />
        <ExampleQuestions onPick={handleAsk} />

        <div className="status">
          {loading && (
            <>
              <span className="spinner" /> Thinking… (first request after
              idle may take ~30s to wake server)
            </>
          )}
          {!loading && error && <span className="status-error">{error}</span>}
          {!loading && !error && answer && (
            fromCache ? (
              <span className="cache-badge">⚡ Cached</span>
            ) : (
              <span className="status-generated">✨ Generated</span>
            )
          )}
        </div>

        {answer && <AnswerCard answer={answer} sources={sources} />}
      </div>

      <footer>Project Aether &bull; High-Performance Event-Driven RAG Engine</footer>
    </main>
  )
}
