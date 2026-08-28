import { useEffect, useRef, useState } from 'react'
import { api } from './api'

const TERMINAL = ['SUCCEEDED', 'FAILED', 'CANCELLED']
const POLL_MS = 1000

// Component 5 of 5. Watch an analysis run, then show what it produced.
//
// WHY setTimeout AND NOT setInterval
// ----------------------------------
// setInterval fires on a schedule regardless of whether the previous request came
// back. One slow response and requests stack up, arrive out of order, and the event
// list flickers backwards. Chaining a setTimeout after each response means there is
// never more than one request in flight, and the gap is measured from the *reply*
// rather than from the send — which is what "poll every second" was always meant to
// mean.
//
// WHY THE CURSOR
// --------------
// Each poll asks for events after the highest id already seen, so a hundred-step M5
// analysis transfers each event exactly once instead of re-sending the whole list
// every second.
export default function Result({ analysisId }) {
  const [events, setEvents] = useState([])
  const [status, setStatus] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [error, setError] = useState(null)
  const cursor = useRef(0)

  useEffect(() => {
    if (!analysisId) return

    // A ref, not state: this must be readable by the in-flight timeout without
    // re-triggering the effect, and it must reset when the id changes.
    cursor.current = 0
    setEvents([])
    setAnalysis(null)
    setError(null)
    setStatus('PENDING')

    let stopped = false
    let timer = null

    async function poll() {
      if (stopped) return
      try {
        const page = await api.getEvents(analysisId, cursor.current)
        if (stopped) return

        if (page.events.length > 0) {
          cursor.current = page.next_after
          setEvents((previous) => [...previous, ...page.events])
        }
        setStatus(page.status)

        if (TERMINAL.includes(page.status)) {
          // Only now fetch the full row. The result payload can be large, and asking
          // for it on every poll would re-transfer it for no reason.
          const full = await api.getAnalysis(analysisId)
          if (!stopped) setAnalysis(full)
          return
        }
      } catch (err) {
        if (!stopped) setError(err.message)
      }
      timer = setTimeout(poll, POLL_MS)
    }

    poll()

    // Runs when the id changes or the component unmounts. Without it, switching
    // analyses leaves the old poll running and writing into unmounted state.
    return () => {
      stopped = true
      if (timer) clearTimeout(timer)
    }
  }, [analysisId])

  if (!analysisId) return null

  const running = status && !TERMINAL.includes(status)

  return (
    <section>
      <h2>5. Result</h2>
      <p>
        <b>status: {status}</b>{' '}
        {running && (
          <button onClick={() => api.cancelAnalysis(analysisId).catch((e) => setError(e.message))}>
            cancel
          </button>
        )}
      </p>
      {error && <p role="alert">{error}</p>}

      <h3>events</h3>
      <ol>
        {events.map((e) => (
          <li key={e.id}>
            <code>{e.kind}</code> — {e.message}
          </li>
        ))}
      </ol>

      {analysis?.error && (
        <>
          <h3>error</h3>
          <pre>{analysis.error}</pre>
        </>
      )}

      {analysis?.result && <ResultBody result={analysis.result} />}
    </section>
  )
}

function ResultBody({ result }) {
  return (
    <>
      <h3>answer</h3>
      <p style={{ whiteSpace: 'pre-wrap' }}>{result.answer}</p>

      {result.table?.rows?.length > 0 && (
        <>
          <h3>evidence</h3>
          <table border="1" cellPadding="4">
            <thead>
              <tr>
                {result.table.columns.map((c) => (
                  <th key={c}>{c}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.table.rows.map((row, i) => (
                <tr key={i}>
                  {row.map((cell, j) => (
                    <td key={j}>{cell === null ? '—' : String(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}

      {result.chart && <Chart chart={result.chart} />}

      <h3>how it got there</h3>
      <ol>
        {result.steps.map((s, i) => (
          <li key={i}>
            <code>{s.tool}</code> ({s.duration_ms} ms) — {s.ok ? s.summary : `FAILED: ${s.error}`}
            <br />
            <small>{JSON.stringify(s.arguments)}</small>
          </li>
        ))}
      </ol>
      <p>
        <small>engine: {result.engine}</small>
      </p>
    </>
  )
}

// A chart drawn out of text. The backend returns a chart SPEC — data plus axis
// metadata, never a rendered image (D-021) — so a real charting library at M6 is a
// swap of this component and nothing else. Until then, block characters prove the
// spec carries everything a renderer needs.
function Chart({ chart }) {
  const spec = chart.chart ?? chart
  const points = spec.data ?? []
  if (points.length === 0) return null
  const max = Math.max(...points.map((p) => Math.abs(Number(p.y) || 0)), 1)

  return (
    <>
      <h3>{spec.title ?? 'chart'}</h3>
      <pre>
        {points
          .map((p) => {
            const width = Math.round((Math.abs(Number(p.y) || 0) / max) * 40)
            return `${String(p.x).padEnd(14).slice(0, 14)} ${'█'.repeat(width).padEnd(40)} ${p.y}`
          })
          .join('\n')}
      </pre>
      <p>
        <small>
          {spec.x?.label} vs {spec.y?.label} — {points.length} point(s)
        </small>
      </p>
    </>
  )
}
