import { formatNumber } from '../format'

// Five chart types, chosen by the backend from the shape of the result.
//
// WHY THE BACKEND PICKS AND NOT THIS COMPONENT
// --------------------------------------------
// The type is a function of the data — how many rows, which columns are numeric,
// whether the x-axis is time — and the backend is holding all of that when it builds
// the spec. Deciding here would mean re-deriving facts that were already known, in a
// place with less information.
//
// WHY NOT A CHARTING LIBRARY
// --------------------------
// Recharts and friends are 100-400 kB to draw five chart types on datasets capped at
// 50 rows. Hand-drawn SVG is a few hundred lines, has no version to keep up with, and
// — the part that matters — means every visual decision here is one we made on purpose
// rather than inherited from a default. When a library becomes worth it (zoom, tooltips
// with crosshairs, brushing) it replaces this file and nothing else, because the spec
// is data and always was (D-021).
//
// SCALED FROM ZERO, ALWAYS. Starting an axis at the smallest value makes a 3%
// difference look like a landslide, which is the chart equivalent of the fabricated
// number this whole project exists to prevent.
export default function Chart({ chart }) {
  const spec = chart?.chart ?? chart
  const points = spec?.data ?? []
  if (points.length === 0) return null

  const Renderer =
    { line: LineChart, pie: PieChart, scatter: ScatterChart, histogram: Bars }[spec.type] ?? Bars

  return (
    <figure className="chart">
      <figcaption className="chart-head">
        <span className="chart-title">{spec.title || 'Chart'}</span>
        <span>
          {spec.type} · {spec.y?.label || 'value'}
        </span>
      </figcaption>
      <Renderer spec={spec} points={points} />
    </figure>
  )
}

// ── bar / histogram ──────────────────────────────────────────────────────────
// Horizontal bars, because category labels are words: vertical bars force the labels
// to be rotated, truncated, or dropped, and a chart whose labels you cannot read is a
// picture of some numbers rather than a chart.

function Bars({ spec, points }) {
  const values = points.map((p) => Number(p.y) || 0)
  const hasNegative = values.some((v) => v < 0)
  const scale = Math.max(...values.map(Math.abs), 1)

  return (
    <div className="chart-rows">
      {points.map((point, index) => {
        const value = Number(point.y) || 0
        return (
          // eslint-disable-next-line react/no-array-index-key -- labels may repeat
          <Row key={index} label={String(point.x)}>
            <div className="chart-track">
              <div
                className={hasNegative && value < 0 ? 'chart-bar is-negative' : 'chart-bar'}
                style={{ width: `${Math.max((Math.abs(value) / scale) * 100, 0.8)}%` }}
              />
            </div>
            <div className="chart-value">{formatNumber(value)}</div>
          </Row>
        )
      })}
    </div>
  )
}

// A fragment, not a wrapper: the cells belong to the parent grid, and a div around
// them would break the column alignment that is the point of using a grid.
function Row({ label, children }) {
  return (
    <>
      <div className="chart-label" title={label}>
        {label}
      </div>
      {children}
    </>
  )
}

// ── line ─────────────────────────────────────────────────────────────────────

function LineChart({ spec, points }) {
  const values = points.map((p) => Number(p.y) || 0)
  const max = Math.max(...values, 0)
  const min = Math.min(...values, 0) // zero is always in view; see the note above
  const span = max - min || 1

  const width = 100
  const height = 100
  const x = (i) => (points.length === 1 ? width / 2 : (i / (points.length - 1)) * width)
  const y = (v) => height - ((v - min) / span) * height

  const path = values.map((v, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(v)}`).join(' ')
  const area = `${path} L ${x(values.length - 1)} ${height} L ${x(0)} ${height} Z`

  // Labels thin out rather than overlap: twelve months fit, fifty days do not, and a
  // smeared axis is less use than a sparse one.
  const every = Math.ceil(points.length / 8)

  return (
    <div className="plot">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="plot-svg">
        <path d={area} className="line-area" />
        <path d={path} className="line-path" vectorEffect="non-scaling-stroke" />
        {points.map((point, i) => (
          // eslint-disable-next-line react/no-array-index-key -- x values may repeat
          <circle key={i} cx={x(i)} cy={y(values[i])} r="1.2" className="line-dot">
            <title>{`${point.x}: ${formatNumber(values[i])}`}</title>
          </circle>
        ))}
      </svg>
      <div className="plot-axis-y">
        <span>{formatNumber(max)}</span>
        <span>{formatNumber(min)}</span>
      </div>
      <div className="plot-axis-x">
        {points.map((point, i) =>
          i % every === 0 ? (
            // eslint-disable-next-line react/no-array-index-key
            <span key={i} style={{ left: `${x(i)}%` }} title={String(point.x)}>
              {String(point.x)}
            </span>
          ) : null,
        )}
      </div>
      <span className="plot-label">{spec.x?.label}</span>
    </div>
  )
}

// ── pie ──────────────────────────────────────────────────────────────────────

// Distinct enough to tell apart at a glance, and ordered so the largest slice gets the
// strongest colour. Capped at eight upstream, which is why there are eight.
const SLICE_COLOURS = [
  'var(--accent)',
  '#22a2a2',
  '#e0813c',
  '#8b5cf6',
  '#3aa76d',
  '#d15b8f',
  '#c9a227',
  '#6b7280',
]

function PieChart({ points }) {
  const values = points.map((p) => Math.max(Number(p.y) || 0, 0))
  const total = values.reduce((sum, v) => sum + v, 0)
  if (total <= 0) return null

  // A donut rather than a filled pie: the hole gives the total somewhere to live, and
  // comparing arc lengths on a ring is easier than comparing wedge areas.
  const radius = 15.9154943 // circumference = 100, so dasharray is literally a percent
  let offset = 25 // start at twelve o'clock instead of three

  return (
    <div className="pie-wrap">
      <svg viewBox="0 0 42 42" className="pie">
        {values.map((value, index) => {
          const percent = (value / total) * 100
          const dash = `${percent} ${100 - percent}`
          const circle = (
            <circle
              // eslint-disable-next-line react/no-array-index-key
              key={index}
              cx="21"
              cy="21"
              r={radius}
              fill="transparent"
              stroke={SLICE_COLOURS[index % SLICE_COLOURS.length]}
              strokeWidth="7"
              strokeDasharray={dash}
              strokeDashoffset={offset}
            >
              <title>{`${points[index].x}: ${formatNumber(value)} (${percent.toFixed(1)}%)`}</title>
            </circle>
          )
          offset -= percent
          return circle
        })}
      </svg>
      <ul className="legend">
        {points.map((point, index) => (
          // eslint-disable-next-line react/no-array-index-key
          <li key={index}>
            <span
              className="swatch"
              style={{ background: SLICE_COLOURS[index % SLICE_COLOURS.length] }}
            />
            <span className="legend-label" title={String(point.x)}>
              {String(point.x)}
            </span>
            <span className="legend-value">
              {((values[index] / total) * 100).toFixed(1)}%
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

// ── scatter ──────────────────────────────────────────────────────────────────

function ScatterChart({ spec, points }) {
  const xs = points.map((p) => Number(p.x) || 0)
  const ys = points.map((p) => Number(p.y) || 0)
  const xMin = Math.min(...xs)
  const xMax = Math.max(...xs)
  const yMin = Math.min(...ys)
  const yMax = Math.max(...ys)
  const xSpan = xMax - xMin || 1
  const ySpan = yMax - yMin || 1

  return (
    <div className="plot">
      <svg viewBox="0 0 100 100" preserveAspectRatio="none" className="plot-svg">
        {points.map((point, i) => (
          <circle
            // eslint-disable-next-line react/no-array-index-key
            key={i}
            cx={((xs[i] - xMin) / xSpan) * 100}
            cy={100 - ((ys[i] - yMin) / ySpan) * 100}
            r="1.4"
            className="scatter-dot"
          >
            <title>{`${formatNumber(xs[i])}, ${formatNumber(ys[i])}`}</title>
          </circle>
        ))}
      </svg>
      <div className="plot-axis-y">
        <span>{formatNumber(yMax)}</span>
        <span>{formatNumber(yMin)}</span>
      </div>
      <div className="plot-axis-x is-range">
        <span>{formatNumber(xMin)}</span>
        <span>{formatNumber(xMax)}</span>
      </div>
      <span className="plot-label">
        {spec.x?.label} vs {spec.y?.label}
      </span>
    </div>
  )
}
