// Turning machine values into things a person reads without effort.
//
// This exists because the first styled build still showed `8187806.363998199` in a
// table cell. That number is correct, and it is unreadable: sixteen digits of float
// noise where two decimal places was the whole meaning. Formatting is not decoration —
// an unreadable true number is not much better than a readable false one.

// A float that is really an integer should not grow ".00", and a genuine decimal
// should not show more precision than the data has.
export function formatNumber(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return String(value)
  if (Number.isInteger(value)) return value.toLocaleString('en-US')

  const magnitude = Math.abs(value)
  // Small numbers are usually rates, shares or unit prices, where the digits after the
  // point are the point. Large ones are totals, where they are noise.
  const decimals = magnitude >= 1000 ? 2 : magnitude >= 1 ? 2 : 4
  return value.toLocaleString('en-US', {
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals,
  })
}

// 541909 -> "542k". Used where the exact figure is not the point, such as a dataset
// row count in a sidebar that has to fit in 260 pixels.
export function formatCompact(value) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  if (value < 1000) return String(value)
  return new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(
    value,
  )
}

// Elapsed time as a person says it: "8s", "1m 12s", "3m 04s".
export function formatDuration(seconds) {
  const whole = Math.max(0, Math.floor(seconds))
  if (whole < 60) return `${whole}s`
  const minutes = Math.floor(whole / 60)
  const rest = whole % 60
  return `${minutes}m ${String(rest).padStart(2, '0')}s`
}

// Tool timings arrive in milliseconds and span five orders of magnitude — 0.04 ms for
// a cached lookup, 157,000 ms for a model call.
export function formatMillis(ms) {
  if (typeof ms !== 'number' || !Number.isFinite(ms)) return ''
  if (ms < 1) return '<1 ms'
  if (ms < 1000) return `${Math.round(ms)} ms`
  return `${(ms / 1000).toFixed(ms < 10000 ? 1 : 0)} s`
}

export function isNumeric(value) {
  return typeof value === 'number' && Number.isFinite(value)
}

// A cell as text. `null` becomes an em dash rather than the string "null", because a
// missing value is a fact about the data and "null" is a fact about JavaScript.
export function formatCell(value) {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return formatNumber(value)
  if (typeof value === 'boolean') return value ? 'true' : 'false'
  return String(value)
}

// `sum(revenue)`, `TotalRevenue`, `share_of_total` -> readable column headers without
// destroying names the user's own data supplied.
export function prettyColumn(name) {
  return String(name).replace(/_/g, ' ')
}

// SQL comes back as one long line. Breaking before the major clauses makes it
// skimmable without pulling in a formatter dependency to do it properly.
const CLAUSES = /\s+(FROM|WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|UNION ALL|UNION)\s+/gi

export function formatSql(sql) {
  if (typeof sql !== 'string') return ''
  return sql
    .replace(/\s+/g, ' ')
    .trim()
    .replace(CLAUSES, (_, clause) => `\n${clause.toUpperCase()} `)
}
