const EXAMPLES = [
  { icon: '⚡', label: 'C++ experience?', query: 'What C++ experience does Gabriel have?' },
  { icon: '🚀', label: 'AI/RAG projects?', query: 'What AI and RAG projects has Gabriel built?' },
  { icon: '🎓', label: 'Education?', query: "What is Gabriel's education background?" },
]

export default function ExampleQuestions({ onPick }) {
  return (
    <div className="examples">
      {EXAMPLES.map((ex) => (
        <button
          key={ex.query}
          type="button"
          className="example-btn"
          onClick={() => onPick(ex.query)}
        >
          {ex.icon} {ex.label}
        </button>
      ))}
    </div>
  )
}
