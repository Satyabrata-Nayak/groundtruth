import { useRef, useState } from 'react'
import { api } from '../api'
import { formatCompact } from '../format'
import SchemaPanel from './SchemaPanel'

// Navigation, not a step in a workflow.
//
// The old page numbered upload as "1." and datasets as "2.", which framed the whole
// application as a form to be filled in from the top. It is not: choosing a dataset is
// something you do once and then forget, and the thing you actually came to do — ask a
// question — was pushed to section four, below a nine-column table.
//
// Everything that is context rather than action lives here, permanently visible and
// permanently out of the way.
// `latest_version` on the API is the version NUMBER, not the version object — a
// computed property, not a relationship. The row and column counts live on the entry
// in `versions` that it names. Reading it as an object silently yields undefined and
// renders "undefined rows", which is the kind of bug that survives a screenshot.
function describe(dataset) {
  const versions = dataset.versions ?? []
  const version =
    versions.find((v) => v.version === dataset.latest_version) ?? versions[versions.length - 1]
  if (!version) return 'no versions'
  return `v${version.version} · ${formatCompact(version.row_count)} rows · ${version.column_count} cols`
}

export default function Sidebar({ datasets, selected, onSelect, onUploaded, onDeleted, health }) {
  const [busy, setBusy] = useState(false)
  const [dragging, setDragging] = useState(false)
  const [error, setError] = useState(null)
  const fileInput = useRef(null)

  async function upload(file) {
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      onUploaded(await api.uploadDataset(file, file.name.replace(/\.[^.]+$/, '')))
    } catch (err) {
      setError(err.message)
    } finally {
      setBusy(false)
      if (fileInput.current) fileInput.current.value = ''
    }
  }

  const healthy = health?.status === 'ok' && health?.database

  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark">G</span>
        <span className="brand-name">Ground Truth</span>
        <span className="brand-spacer" />
        <span
          className={`dot ${health ? (healthy ? 'is-ok' : 'is-down') : ''}`}
          title={
            healthy
              ? 'API and database are up'
              : 'API or database unreachable — is uvicorn running?'
          }
        />
      </div>

      <div className="sidebar-scroll">
        {/* Shown only when something is wrong. In the old header this line was always
            present, which made a permanent instruction out of what is only ever
            useful as an alarm. */}
        {health && !healthy && (
          <div className="callout is-error" style={{ margin: '4px 0 10px' }}>
            <span className="callout-icon">✕</span>
            <span>
              API unreachable. Start <code>uvicorn app.api.main:app</code>.
            </span>
          </div>
        )}

        <div className="side-label">Datasets</div>

        {datasets.length === 0 && <p className="ds-meta" style={{ padding: '2px 8px' }}>none yet</p>}

        {datasets.map((dataset) => (
          <div
            key={dataset.id}
            className={`ds ${selected?.id === dataset.id ? 'is-active' : ''}`}
            onClick={() => onSelect(dataset)}
            onKeyDown={(e) => e.key === 'Enter' && onSelect(dataset)}
            role="button"
            tabIndex={0}
          >
            <div className="ds-body">
              <div className="ds-name">{dataset.name}</div>
              <div className="ds-meta">{describe(dataset)}</div>
            </div>
            <button
              type="button"
              className="ds-delete"
              title="Delete this dataset"
              onClick={(e) => {
                e.stopPropagation()
                api.deleteDataset(dataset.id).then(() => onDeleted(dataset.id))
              }}
            >
              ×
            </button>
          </div>
        ))}

        <label
          className={`upload ${dragging ? 'is-over' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault()
            setDragging(false)
            upload(e.dataTransfer.files?.[0])
          }}
        >
          <input
            ref={fileInput}
            type="file"
            accept=".csv,.parquet"
            onChange={(e) => upload(e.target.files?.[0])}
          />
          {busy ? (
            <>
              <span className="spin">◠</span> uploading…
            </>
          ) : (
            'drop a CSV or click to upload'
          )}
        </label>

        {error && <p className="ds-meta" style={{ color: 'var(--danger)', padding: '6px 8px' }}>{error}</p>}

        <SchemaPanel dataset={selected} />
      </div>
    </aside>
  )
}
