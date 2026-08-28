import { useEffect, useRef, useState } from 'react'
import { api } from './api'

const TERMINAL = ['SUCCEEDED', 'FAILED', 'CANCELLED']
const POLL_MS = 1000

// Watching one analysis run.
//
// WHY setTimeout AND NOT setInterval
// ----------------------------------
// An interval fires on schedule regardless of whether the previous request came back.
// One slow response and requests stack up, arrive out of order, and the event list
// flickers backwards. Chaining a setTimeout after each response means there is never
// more than one request in flight and the gap is measured from the REPLY — which is
// what "poll every second" always meant.
//
// WHY THE CURSOR
// --------------
// Each poll asks for events after the highest id already seen, so a long analysis
// transfers each event exactly once rather than re-sending the whole list every second.
//
// WHY A SEPARATE ELAPSED CLOCK
// ----------------------------
// The elapsed timer ticks every second from a local start time, not from poll replies.
// Deriving it from the last event would freeze the display for the two minutes a model
// call takes — which is precisely the stretch where a moving number is the only
// evidence the app has not died.
export function useAnalysis(analysisId) {
  const [events, setEvents] = useState([])
  const [status, setStatus] = useState(null)
  const [analysis, setAnalysis] = useState(null)
  const [error, setError] = useState(null)
  const [elapsed, setElapsed] = useState(0)
  const cursor = useRef(0)

  useEffect(() => {
    if (!analysisId) return undefined

    // Refs, not state: readable by the in-flight timeout without re-triggering the
    // effect, and reset when the id changes.
    cursor.current = 0
    setEvents([])
    setAnalysis(null)
    setError(null)
    setStatus('PENDING')
    setElapsed(0)

    let stopped = false
    let timer = null

    // The elapsed clock ticks once a second, and must STOP when the analysis does.
    // Left running it would re-render every finished turn in the thread once a second
    // forever — invisible with one exchange on screen and increasingly janky with ten,
    // which is exactly the shape this UI was rebuilt to encourage.
    const startedAt = Date.now()
    const clock = setInterval(() => {
      if (!stopped) setElapsed((Date.now() - startedAt) / 1000)
    }, 1000)

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
          // Only now fetch the full row. The result payload can be large and asking
          // for it on every poll would re-transfer it for no reason.
          const full = await api.getAnalysis(analysisId)
          if (!stopped) setAnalysis(full)
          clearInterval(clock)
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
      clearInterval(clock)
    }
  }, [analysisId])

  return {
    events,
    status,
    analysis,
    error,
    elapsed,
    running: Boolean(status) && !TERMINAL.includes(status),
  }
}
