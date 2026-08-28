import { useEffect } from 'react'
import { api } from '../api'
import { useAnalysis } from '../useAnalysis'
import Chart from './Chart'
import DataTable from './DataTable'
import LiveStatus from './LiveStatus'
import Steps from './Steps'

// One exchange: what was asked, and what came back.
//
// The reading order inside the reply is deliberate and is the order a person needs it:
//
//   1. the answer          what you asked for
//   2. any warning         what the system is not sure about  ← above the evidence
//   3. the chart           the shape of it
//   4. the table           the numbers themselves
//   5. how it got there    the SQL, collapsed
//
// The warning sits at position two rather than at the bottom because it qualifies
// everything below it. In the old layout it was a bullet under the prose, styled
// identically to the prose, which is how the single most important safety feature in
// the system came to look like a footnote.
export default function Turn({ turn, onStatusChange }) {
  const { events, status, analysis, error, elapsed, running } = useAnalysis(turn.analysisId)

  // The composer stays disabled while a question is in flight: a second question would
  // queue behind the first on a single worker and appear to hang.
  useEffect(() => {
    onStatusChange?.(turn.analysisId, status)
  }, [turn.analysisId, status, onStatusChange])

  const result = analysis?.result

  return (
    <article className="turn">
      <div className="ask">{turn.question}</div>

      <div className="reply">
        {running && <LiveStatus events={events} elapsed={elapsed} />}

        {error && (
          <Callout kind="error">
            <b>Could not reach the API.</b> {error}
          </Callout>
        )}

        {status === 'FAILED' && analysis?.error && (
          <Callout kind="error">
            <b>This question could not be answered.</b> {analysis.error}
          </Callout>
        )}

        {status === 'CANCELLED' && <p className="muted">Cancelled.</p>}

        {result && (
          <>
            <p className="answer">{result.answer}</p>

            {result.warnings?.map((warning) => (
              <Callout key={warning}>
                <b>Unverified.</b> {warning}
              </Callout>
            ))}

            {result.chart && <Chart chart={result.chart} />}
            <DataTable table={result.table} />
            <Steps
              steps={result.steps}
              engine={result.engine}
              model={result.model}
              thinking={result.thinking}
            />
          </>
        )}

        {running && (
          <div>
            <button
              type="button"
              className="linkish"
              onClick={() => api.cancelAnalysis(turn.analysisId).catch(() => {})}
            >
              stop
            </button>
          </div>
        )}
      </div>
    </article>
  )
}

function Callout({ kind, children }) {
  return (
    <div className={kind === 'error' ? 'callout is-error' : 'callout'} role="status">
      <span className="callout-icon" aria-hidden="true">
        {kind === 'error' ? '✕' : '⚠'}
      </span>
      <span>{children}</span>
    </div>
  )
}
