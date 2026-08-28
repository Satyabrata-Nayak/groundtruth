import { formatMillis, formatSql } from '../format'

// "How it got there" — collapsed by default, and the most important thing in the app
// for anyone deciding whether to believe the answer.
//
// The old version printed `{"sql":"SELECT Quantity, UnitPrice, InvoiceNo, StockCode,
// Description FROM dataset ORDER BY Quantity * UnitPrice DESC LIMIT 10","max_rows":50}`
// on one line. That string is the single strongest piece of evidence the system has —
// it is exactly what a sceptical analyst would ask to see — and it was rendered as
// JSON punctuation. Here the SQL gets a code block with its clauses on their own lines,
// and the remaining arguments become small chips beside it.
//
// Collapsed by default because a correct answer is the common case and the trace is
// then noise. One click away because when it matters, it matters completely.
export default function Steps({ steps, engine, model, thinking, chart }) {
  if (!steps?.length) return null

  return (
    <details className="trace">
      <summary>
        How it got there — {steps.length} step{steps.length === 1 ? '' : 's'}
      </summary>
      <div className="trace-body">
        {steps.map((step, index) => (
          // eslint-disable-next-line react/no-array-index-key -- steps have no id
          <Step key={index} step={step} />
        ))}
        {/* THE CHART BELONGS IN THE TRACE TOO.
            It was missing, and its absence was a real gap rather than a cosmetic one:
            "how it got there" is the page's honesty guarantee, and a chart appearing
            with no entry explaining where it came from is exactly the kind of
            unexplained artefact the section exists to prevent. It has no step because
            nothing called a tool — the type is inferred from the shape of the result —
            so it says that. */}
        {chart?.chart && (
          <div className="step">
            <div className="step-head">
              <span className="tool is-derived">{chart.chart.type} chart</span>
              <span className="step-summary">
                {chart.chart.derived_from
                  ? `inferred from the shape of the result — ${chart.chart.point_count} point(s), no tool call`
                  : `${chart.chart.point_count} point(s)`}
              </span>
            </div>
          </div>
        )}

        {/* Which model answered THIS question, not what the picker is set to now.
            Two answers to the same question can differ for no other reason. */}
        <p className="trace-foot">
          Answered by <code>{engine}</code>
          {model ? (
            <>
              {' '}using <code>{model}</code>
              {thinking === false ? ' with reasoning off' : ''}
            </>
          ) : null}
          . Every figure above came out of DuckDB; the model chose what to compute and
          computed nothing.
        </p>
      </div>
    </details>
  )
}

function Step({ step }) {
  const args = step.arguments || {}
  const { sql, ...rest } = args

  return (
    <div className="step">
      <div className="step-head">
        <span className={step.ok ? 'tool' : 'tool is-failed'}>{step.tool}</span>
        <span className="step-summary">{step.ok ? step.summary : step.error}</span>
        <span className="step-time">{formatMillis(step.duration_ms)}</span>
      </div>

      {sql && <pre className="sql">{formatSql(sql)}</pre>}

      {Object.keys(rest).length > 0 && (
        <div className="args">
          {Object.entries(rest).map(([key, value]) => (
            <span className="arg" key={key}>
              {key}: {typeof value === 'object' ? JSON.stringify(value) : String(value)}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
