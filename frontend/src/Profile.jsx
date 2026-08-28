import { useEffect, useState } from 'react'
import { api } from './api'

// Component 3 of 5. The stored profile for the selected dataset.
//
// These numbers are read back from Postgres, not recomputed: they were calculated
// exactly once at ingest (D-012 — exactly, never estimated) and cost nothing to read.
export default function Profile({ dataset }) {
  const [profile, setProfile] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    if (!dataset) {
      setProfile(null)
      return
    }
    let cancelled = false
    setProfile(null)
    setError(null)
    api
      .getProfile(dataset.id)
      // The `cancelled` guard handles the out-of-order response: click A, click B,
      // and A's slower reply would otherwise land last and overwrite B's profile with
      // the wrong dataset's columns. The cleanup below runs before the next effect.
      .then((p) => !cancelled && setProfile(p))
      .catch((e) => !cancelled && setError(e.message))
    return () => {
      cancelled = true
    }
  }, [dataset])

  if (!dataset) return null
  if (error) return <section><h2>3. Profile</h2><p role="alert">{error}</p></section>
  if (!profile) return <section><h2>3. Profile</h2><p>loading…</p></section>

  return (
    <section>
      <h2>3. Profile</h2>
      <p>
        v{profile.version} — {profile.row_count.toLocaleString()} rows,{' '}
        {profile.column_count} columns, {profile.duplicate_row_count} duplicate rows
      </p>
      <table border="1" cellPadding="4">
        <thead>
          <tr>
            <th>column</th>
            <th>type</th>
            <th>kind</th>
            <th>nulls</th>
            <th>distinct</th>
            <th>min</th>
            <th>max</th>
            <th>mean</th>
            <th>flags</th>
          </tr>
        </thead>
        <tbody>
          {profile.columns.map((c) => (
            <tr key={c.name}>
              <td>{c.name}</td>
              <td>{c.duckdb_type}</td>
              <td>{c.semantic_type}</td>
              <td>
                {c.null_count} ({(c.null_fraction * 100).toFixed(1)}%)
              </td>
              <td>{c.distinct_count ?? '—'}</td>
              <td>{c.min_value ?? '—'}</td>
              <td>{c.max_value ?? '—'}</td>
              {/* Nullish coalescing, not `||`: a genuine mean of 0 is not missing, and
                  `||` would render it as an em dash. The same distinction the schema
                  makes by storing NULL rather than 0 for an inapplicable statistic. */}
              <td>{c.mean_value?.toFixed(2) ?? '—'}</td>
              <td>
                {c.is_constant && 'constant '}
                {c.is_high_cardinality && 'high-cardinality'}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  )
}
