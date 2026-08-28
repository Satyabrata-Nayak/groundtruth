import { useState } from 'react'
import { api } from './api'

// Component 4 of 5. Type a question, get back an analysis id.
//
// This POST returns in milliseconds. It does not wait for an answer — it queues a row
// and hands back its id, and Result polls for what happens next. That separation is
// the entire point of M4: in M5 the work behind this button takes 10-60 seconds, and
// nothing about this component changes.
export default function AskBox({ dataset, onQueued }) {
  const [question, setQuestion] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function submit(event) {
    event.preventDefault()
    if (!dataset || !question.trim()) return
    setBusy(true)
    setError(null)
    try {
      const analysis = await api.createAnalysis(dataset.id, question.trim())
      onQueued(analysis)
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <section>
      <h2>4. Ask</h2>
      {!dataset && <p>select a dataset first.</p>}
      <form onSubmit={submit}>
        <input
          type="text"
          size="60"
          placeholder="e.g. which region has the highest revenue?"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          disabled={!dataset}
        />
        <button type="submit" disabled={!dataset || !question.trim() || busy}>
          {busy ? 'queueing…' : 'ask'}
        </button>
      </form>
      {/* Said plainly rather than hidden. M4 runs a fixed analysis; a UI that implied
          otherwise would be the first place this project told a user something untrue. */}
      <p>
        <small>
          M4 runs a fixed analysis and does not read the question yet — it always
          compares the first usable numeric column across the first usable categorical
          one. The question is stored and answered for real in M5.
        </small>
      </p>
      {error && <p role="alert">could not queue: {error}</p>}
    </section>
  )
}
