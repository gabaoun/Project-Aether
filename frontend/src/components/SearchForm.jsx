export default function SearchForm({ query, onQueryChange, onSubmit, loading }) {
  function handleSubmit(e) {
    e.preventDefault()
    onSubmit(query)
  }

  return (
    <form onSubmit={handleSubmit}>
      <div className="input-wrapper">
        <input
          type="text"
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          placeholder="Ask about Gabriel's experience, AI projects, C++..."
          autoComplete="off"
          maxLength={300}
          required
        />
      </div>
      <button className="btn-send" type="submit" disabled={loading}>
        {loading ? 'Asking…' : 'Ask'}
      </button>
    </form>
  )
}
