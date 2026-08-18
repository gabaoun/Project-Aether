import { useState } from 'react'
import { parseMarkdown } from '../utils/parseMarkdown.js'

export default function AnswerCard({ answer, sources }) {
  const [copied, setCopied] = useState(false)

  function handleCopy() {
    navigator.clipboard.writeText(answer).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  return (
    <div className="answer-card">
      <div className="answer-header">
        <span className="answer-title">Answer</span>
        <button className="btn-copy" type="button" onClick={handleCopy}>
          {copied ? 'Copied!' : 'Copy'}
        </button>
      </div>

      <div
        className="answer-body"
        dangerouslySetInnerHTML={{ __html: parseMarkdown(answer) }}
      />

      {sources.length > 0 && (
        <div className="sources-wrapper">
          <div className="sources-title">Sources</div>
          <div className="source-chips">
            {sources.map((src, i) => (
              <span key={`${src}-${i}`} className="source-chip">{src}</span>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
