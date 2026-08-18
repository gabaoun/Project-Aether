import { useCallback, useState } from 'react'

/**
 * Encapsulates the /query request/response cycle: loading state, the
 * parsed answer, its source nodes, cache status, and any error message.
 */
export function useAetherQuery() {
  const [answer, setAnswer] = useState(null)
  const [sources, setSources] = useState([])
  const [fromCache, setFromCache] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const ask = useCallback(async (query) => {
    setLoading(true)
    setError(null)
    setAnswer(null)
    setSources([])

    try {
      const res = await fetch('/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      const data = await res.json()

      if (!res.ok) {
        setError(data.detail || `Error ${res.status}`)
        return
      }

      setAnswer(data.answer)
      setSources(data.source_nodes || [])
      setFromCache(Boolean(data.from_cache))
    } catch (err) {
      setError('Could not reach the API. Server may be waking up, please retry shortly.')
    } finally {
      setLoading(false)
    }
  }, [])

  return { ask, answer, sources, fromCache, loading, error }
}
