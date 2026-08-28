import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from './api'
import Composer from './components/Composer'
import Empty from './components/Empty'
import Sidebar from './components/Sidebar'
import Turn from './components/Turn'

// The shell: a sidebar of context, a thread of exchanges, a composer at the bottom.
//
// WHY A THREAD AND NOT A RESULT PANEL
// -----------------------------------
// The previous layout had one result area that the next question overwrote. That is
// the wrong shape twice over. It is wrong now, because comparing an answer with the
// follow-up that refined it means scrolling between two screenshots. And it is wrong
// for what comes next: memory turns a sequence of questions into a conversation, and a
// UI that only ever holds one exchange has nowhere to put the second.
//
// So `turns` is an append-only array and the thread renders all of it. Older exchanges
// slide up as new ones arrive, which is both the familiar chat behaviour and exactly
// the structure a memory feature needs — with no memory in it yet.
//
// WHY STATE LIVES HERE AND NOT IN A STORE
// ---------------------------------------
// There are four facts in the whole application: which datasets exist, which is
// selected, the list of turns, and whether one is running. Reaching for Redux or
// Zustand at this size is more machinery than the thing it manages.
export default function App() {
  const [datasets, setDatasets] = useState([])
  const [selected, setSelected] = useState(null)
  const [turns, setTurns] = useState([])
  const [health, setHealth] = useState(null)
  const [runningId, setRunningId] = useState(null)
  const bottom = useRef(null)

  const refresh = useCallback(async () => {
    const list = await api.listDatasets()
    setDatasets(list)
    return list
  }, [])

  useEffect(() => {
    refresh().catch(() => setDatasets([]))
    api
      .health()
      .then(setHealth)
      .catch(() => setHealth({ status: 'unreachable', database: false }))
  }, [refresh])

  // Pick the only dataset automatically. Making someone click the single item in a
  // list before they are allowed to type is a step that exists purely to be completed.
  useEffect(() => {
    if (!selected && datasets.length > 0) setSelected(datasets[0])
  }, [datasets, selected])

  // Newest exchange into view whenever one is added. `smooth` in CSS rather than here
  // so `prefers-reduced-motion` can turn it off without a JS branch.
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' })
  }, [turns.length])

  async function ask(question) {
    if (!selected) return
    try {
      const analysis = await api.createAnalysis(selected.id, question)
      setRunningId(analysis.id)
      setTurns((current) => [...current, { question, analysisId: analysis.id }])
    } catch (err) {
      setTurns((current) => [...current, { question, error: err.message }])
    }
  }

  // A turn reports its own status up so the composer knows whether the single worker
  // is busy. Stable identity via useCallback, or every poll would re-run the effect
  // that calls it.
  const handleStatus = useCallback((analysisId, status) => {
    if (['SUCCEEDED', 'FAILED', 'CANCELLED'].includes(status)) {
      setRunningId((current) => (current === analysisId ? null : current))
    }
  }, [])

  function selectDataset(dataset) {
    setSelected(dataset)
    // A thread is about one dataset. Carrying answers about `retail` into a session
    // about `sales` would put two datasets' numbers on one page with nothing saying
    // which was which.
    setTurns([])
    setRunningId(null)
  }

  return (
    <div className="app">
      <Sidebar
        datasets={datasets}
        selected={selected}
        health={health}
        onSelect={selectDataset}
        onUploaded={async (dataset) => {
          const list = await refresh()
          selectDataset(list.find((d) => d.id === dataset.id) ?? dataset)
        }}
        onDeleted={(id) => {
          setDatasets((current) => current.filter((d) => d.id !== id))
          if (selected?.id === id) selectDataset(null)
        }}
      />

      <main className="main">
        <div className="thread">
          <div className="thread-inner">
            {turns.length === 0 ? (
              <Empty dataset={selected} onPick={ask} />
            ) : (
              turns.map((turn) =>
                turn.error ? (
                  <article className="turn" key={turn.question + turn.error}>
                    <div className="ask">{turn.question}</div>
                    <div className="reply">
                      <div className="callout is-error">
                        <span className="callout-icon">✕</span>
                        <span>
                          <b>Could not queue this question.</b> {turn.error}
                        </span>
                      </div>
                    </div>
                  </article>
                ) : (
                  <Turn key={turn.analysisId} turn={turn} onStatusChange={handleStatus} />
                ),
              )
            )}
            <div ref={bottom} />
          </div>
        </div>

        <Composer dataset={selected} busy={Boolean(runningId)} onAsk={ask} />
      </main>
    </div>
  )
}
