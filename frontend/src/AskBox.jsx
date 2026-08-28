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
      {/* An honest expectation, set before the wait rather than after it. A local 4B
          model reasons for 30-60 seconds per turn, and a progress trail that starts
          moving immediately is only reassuring if the user knows what it is waiting
          for. */}
      <p>
        <small>
          A local model answers this by running queries against your data — it computes
          every number rather than recalling one. Expect 30 seconds to three minutes;
          each step appears below as it happens.
        </small>
      </p>
      {error && <p role="alert">could not queue: {error}</p>}
    </section>
  )
}
