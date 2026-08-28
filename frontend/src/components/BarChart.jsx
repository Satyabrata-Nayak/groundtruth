import { formatNumber } from '../format'

// Bars made of CSS, not of block characters.
//
// The previous version drew `█`.repeat(n) into a <pre>. On the retail data that
// produced a solid black smear: one bar at 80,995 next to six bars at 1, in a
// monospace font, is 40 filled cells beside 0 filled cells and no visible axis. It
// read as a rendering bug rather than as a chart, which is worse than showing nothing.
//
// Percentage-width divs are responsive for free — no viewBox arithmetic, no resize
// observer — and a real grid puts the labels, bars and values in three aligned columns
// so the eye can travel down any one of them.
//
// SCALED FROM ZERO, ALWAYS. Starting the axis at the smallest value makes a 3%
// difference look like a landslide, which is the chart equivalent of the fabricated
// number this whole project exists to prevent.
export default function BarChart({ chart }) {
  const spec = chart?.chart ?? chart
  const points = spec?.data ?? []
  if (points.length === 0) return null

  const values = points.map((point) => Number(point.y) || 0)
  const hasNegative = values.some((value) => value < 0)
  // With negatives present the reference is the widest swing either way, so a -50 and
  // a +50 draw the same length and the sign is what distinguishes them.
  const scale = Math.max(...values.map(Math.abs), 1)

  return (
    <div className="chart">
      <div className="chart-head">
        <span className="chart-title">{spec.title || 'Chart'}</span>
        <span>
          {spec.y?.label || 'value'}
          {spec.derived_from ? ' · from the result above' : ''}
        </span>
      </div>

      <div className="chart-rows">
        {points.map((point, index) => {
          const value = Number(point.y) || 0
          const width = `${Math.max((Math.abs(value) / scale) * 100, 0.8)}%`
          return (
            // eslint-disable-next-line react/no-array-index-key -- labels may repeat
            <Row key={index} label={String(point.x)} width={width} negative={hasNegative && value < 0}>
              {formatNumber(value)}
            </Row>
          )
        })}
      </div>
    </div>
  )
}

// A fragment rather than a wrapper element: the three cells belong to the parent grid,
// and a div around them would break the column alignment that is the entire point.
function Row({ label, width, negative, children }) {
  return (
    <>
      <div className="chart-label" title={label}>
        {label}
      </div>
      <div className="chart-track">
        <div
          className={negative ? 'chart-bar is-negative' : 'chart-bar'}
          style={{ width }}
        />
      </div>
      <div className="chart-value">{children}</div>
    </>
  )
}
