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
// localStorage access throws rather than returning null in a private window or with
// site data blocked, so both sides are guarded. A remembered dropdown is not worth a
// blank page.
function read(key) {
  try {
    return window.localStorage.getItem(key) ?? null
  } catch {
    return null
  }
}

function write(key, value) {
  try {
    if (value === null) window.localStorage.removeItem(key)
    else window.localStorage.setItem(key, value)
  } catch {
    // nothing to do: the preference simply will not survive this session
  }
}

export default function App() {
  const [datasets, setDatasets] = useState([])
  const [selected, setSelected] = useState(null)
  const [turns, setTurns] = useState([])
  const [health, setHealth] = useState(null)
  const [runningId, setRunningId] = useState(null)
  // The thread the next question continues. Held in state rather than persisted:
  // a conversation is scoped to one dataset and one sitting, and reloading the page
  // to a thread whose earlier answers are no longer on screen would be worse than
  // starting fresh.
  const [conversationId, setConversationId] = useState(null)
  const [models, setModels] = useState([])
  // The chosen model and reasoning flag persist across reloads, because a preference
  // you have to re-make every visit is not a preference. Wrapped in try/catch: a
  // private window or blocked site data throws on access rather than returning null,
  // and a chat app should not fail to start over a remembered dropdown.
  const [model, setModel] = useState(() => read('gt.model'))
  const [thinking, setThinking] = useState(() => (read('gt.thinking') === 'false' ? false : null))
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
    api
      .listModels()
      .then(setModels)
      .catch(() => setModels([]))
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
      const analysis = await api.createAnalysis(selected.id, question, {
        model,
        thinking,
        conversationId,
      })
      // Captured from the FIRST answer and sent with every one after it. This single
      // line is what turns a list of independent questions into a conversation.
      setConversationId(analysis.conversation_id ?? conversationId)
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
    // A thread belongs to one dataset. Carrying it across would let a follow-up about
    // `sensors` be answered with a fact established about `retail`.
    setConversationId(null)
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

        <Composer
          dataset={selected}
          busy={Boolean(runningId)}
          onAsk={ask}
          models={models}
          model={model}
          thinking={thinking}
          onModel={(name) => {
            setModel(name)
            write('gt.model', name)
          }}
          onThinking={(value) => {
            setThinking(value)
            write('gt.thinking', value === false ? 'false' : null)
          }}
        />
      </main>
    </div>
  )
}
