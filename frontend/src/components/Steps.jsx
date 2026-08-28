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
export default function Steps({ steps, engine }) {
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
        <p className="trace-foot">
          Answered by <code>{engine}</code>. Every figure above came out of DuckDB; the
          model chose what to compute and computed nothing.
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
