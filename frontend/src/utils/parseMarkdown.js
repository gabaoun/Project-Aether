// Minimal markdown-lite renderer for RAG answers: bold, inline code, and
// list/paragraph blocks. Not a full markdown parser, matches the exact
// subset the model output actually uses.
export function parseMarkdown(text) {
  if (!text) return ''

  let html = text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')

  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>')

  const lines = html.split('\n')
  let inList = false
  let result = ''

  lines.forEach((line) => {
    const trimmed = line.trim()
    if (trimmed.startsWith('- ') || /^\d+\.\s/.test(trimmed)) {
      if (!inList) {
        result += '<ul>'
        inList = true
      }
      const itemContent = trimmed.replace(/^(-\s*|\d+\.\s*)/, '')
      result += `<li>${itemContent}</li>`
    } else {
      if (inList) {
        result += '</ul>'
        inList = false
      }
      if (trimmed.length > 0) {
        result += `<p>${line}</p>`
      }
    }
  })

  if (inList) result += '</ul>'
  return result
}
