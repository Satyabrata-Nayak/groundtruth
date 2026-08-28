import { useCallback, useEffect, useState } from 'react'
import { api } from './api'
import AskBox from './AskBox'
import DatasetList from './DatasetList'
import Profile from './Profile'
import Result from './Result'
import Upload from './Upload'

// The whole app: five components, one piece of shared state each way.
//
// State lives here rather than in a store because there are three facts in total —
// which datasets exist, which is selected, which analysis is being watched. Reaching
// for Redux or Zustand at this size would be more code than the thing it manages.
export default function App() {
  const [datasets, setDatasets] = useState([])
  const [selected, setSelected] = useState(null)
  const [analysisId, setAnalysisId] = useState(null)
  const [health, setHealth] = useState(null)

  const refresh = useCallback(async () => {
    const list = await api.listDatasets()
    setDatasets(list)
    return list
  }, [])

  useEffect(() => {
    refresh().catch(() => setDatasets([]))
    // A visible answer to "is anything actually running?". Without it, the first
    // failure of a forgotten `docker compose up` looks like a frontend bug.
    api.health().then(setHealth).catch(() => setHealth({ status: 'unreachable', database: false }))
  }, [refresh])

  async function handleUploaded(dataset) {
    const list = await refresh()
    setSelected(list.find((d) => d.id === dataset.id) ?? dataset)
    setAnalysisId(null)
  }

  function handleDeleted(id) {
    setDatasets((current) => current.filter((d) => d.id !== id))
    if (selected?.id === id) {
      setSelected(null)
      setAnalysisId(null)
    }
  }

  return (
    <main>
      <h1>AI Data Analyser — M4</h1>
      <p>
        <small>
          api: {health ? `${health.status} (database ${health.database ? 'up' : 'DOWN'})` : '…'} —
          needs <code>uvicorn app.api.main:app</code> and <code>python -m app.worker</code> running.
        </small>
      </p>
      <hr />

      <Upload onUploaded={handleUploaded} />
      <hr />

      <DatasetList
        datasets={datasets}
        selectedId={selected?.id ?? null}
        onSelect={(d) => {
          setSelected(d)
          setAnalysisId(null)
        }}
        onDeleted={handleDeleted}
      />
      <hr />

      <Profile dataset={selected} />
      <hr />

      <AskBox dataset={selected} onQueued={(a) => setAnalysisId(a.id)} />
      <hr />

      <Result analysisId={analysisId} />
    </main>
  )
}
