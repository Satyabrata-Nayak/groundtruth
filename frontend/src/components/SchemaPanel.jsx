import { useEffect, useState } from 'react'
import { api } from '../api'
import { formatCompact, formatNumber } from '../format'

// The dataset's shape, as a reference rather than as a report.
//
// The old version rendered the full profile as a nine-column table in the middle of
// the page: column, type, kind, nulls, distinct, min, max, mean, flags. On the retail
// data five of those columns were an em dash for every VARCHAR, and `flags` was empty
// for all eight rows — so the widest, most visually dominant element on the page was
// mostly punctuation, sitting directly above the thing people came to use.
//
// What a person actually needs while writing a question is: what are the columns
// called, which ones are numbers, and which ones have missing data. That is three
// facts, and they fit in a sidebar. The rest is real and stays one click away.
export default function SchemaPanel({ dataset }) {
  const [profile, setProfile] = useState(null)
  const [error, setError] = useState(null)
  const [full, setFull] = useState(false)

  useEffect(() => {
    if (!dataset) {
      setProfile(null)
      return undefined
    }
    let cancelled = false
    setProfile(null)
    setError(null)
    api
      .getProfile(dataset.id)
      // The guard handles the out-of-order response: click A, click B, and A's slower
      // reply would otherwise land last and show the wrong dataset's columns.
      .then((p) => !cancelled && setProfile(p))
      .catch((e) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [dataset])

  if (!dataset) return null

  return (
    <>
      <div className="side-label">
        <span>Columns</span>
        {profile && (
          <button type="button" className="linkish" onClick={() => setFull((v) => !v)}>
            {full ? 'less' : 'stats'}
          </button>
        )}
      </div>

      {error && <p className="ds-meta" style={{ padding: '2px 8px' }}>{error}</p>}
      {!profile && !error && <p className="ds-meta" style={{ padding: '2px 8px' }}>loading…</p>}

      {profile?.columns.map((column) => (
        <div className="col" key={column.name} title={`${column.duckdb_type} · ${column.name}`}>
          <span className="col-name">{column.name}</span>
          {/* Only nulls worth mentioning get a badge. A column that is 0.0% null does
              not need to announce that on every render. */}
          {column.null_fraction > 0.005 && (
            <span className="col-null">{Math.round(column.null_fraction * 100)}%∅</span>
          )}
          <span className={`kind ${column.semantic_type === 'numeric' ? 'is-numeric' : ''}`}>
            {column.semantic_type === 'numeric' ? '123' : 'abc'}
          </span>
        </div>
      ))}

      {full && profile && (
        <div style={{ padding: '10px 8px 0' }}>
          <div className="ds-meta" style={{ lineHeight: 1.7 }}>
            {formatCompact(profile.row_count)} rows · {profile.column_count} columns
            <br />
            {formatNumber(profile.duplicate_row_count)} duplicate rows
          </div>
          <div style={{ marginTop: 10 }}>
            {profile.columns.map((column) => (
              <div className="col" key={column.name} style={{ display: 'block', paddingTop: 6 }}>
                <div style={{ fontWeight: 600, color: 'var(--ink-soft)' }}>{column.name}</div>
                <div className="ds-meta">
                  {column.duckdb_type} · {formatCompact(column.distinct_count)} distinct
                  {column.mean_value !== null && column.mean_value !== undefined
                    ? ` · mean ${formatNumber(column.mean_value)}`
                    : ''}
                  {column.is_constant ? ' · constant' : ''}
                  {column.is_high_cardinality ? ' · identifier-like' : ''}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  )
}
