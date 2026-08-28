// Turning the event log into "what is happening right now".
//
// THE INSIGHT THIS FILE IS BUILT ON
// --------------------------------
// Events record what ALREADY HAPPENED. `MODEL_CALL` is written after the call returns,
// which means that while the model is actually thinking, NO EVENT ARRIVES AT ALL. A UI
// that renders the event list is therefore silent for exactly the stretch where the
// user most needs to know something is happening.
//
// So the current phase is derived from the state AFTER the last event, not from the
// last event itself.
//
// WHY THERE IS NO LONGER A "STEP N OF 6"
// --------------------------------------
// There was, and it lied. `step` came from the newest MODEL_CALL, which is written
// when a call FINISHES — so during the long second call the counter showed the step
// that had already completed, and sat on "step 1 of 6" for two minutes while step 2
// ran. Reporting a completed step as the current one is worse than reporting nothing.
//
// The loop is also no longer step-shaped. It is: plan (with tools), run everything the
// model asked for, write the answer (without tools). Naming the phase is both honest
// and more informative than a counter over a budget the user does not care about.

const TOOL_LABELS = {
  inspect_schema: 'Reading the schema',
  profile_column: 'Profiling a column',
  execute_sql: 'Running a query',
  compare_groups: 'Comparing groups',
  correlation: 'Measuring a correlation',
  create_chart: 'Building a chart',
}

function phaseAfter(event) {
  if (!event) return { label: 'Starting' }

  const payload = event.payload || {}

  switch (event.kind) {
    case 'QUEUED':
      return { label: 'Waiting for a worker' }

    case 'CLAIMED':
    case 'RECLAIMED':
      return { label: 'Reading your data' }

    // Announced but not yet reported back: it is running.
    case 'TOOL_CALL':
      return { label: TOOL_LABELS[payload.tool] || `Running ${payload.tool || 'a tool'}` }

    // A tool just finished. In the current loop, a successful query is followed
    // immediately by the answer call — the tools are taken away as soon as there is
    // something to answer from, because a model holding tools re-argues whether to use
    // them and takes four times as long doing it.
    case 'TOOL_RESULT':
      return { label: 'Writing the answer', detail: event.message }

    case 'MODEL_CALL':
      if (/writing the answer/i.test(payload.phase || event.message)) {
        return { label: 'Checking the figures' }
      }
      // The model has named the tools it wants; they are about to run.
      return /ready to answer/i.test(event.message)
        ? { label: 'Writing the answer' }
        : { label: 'Running the queries' }

    case 'NOTE':
      return /answer written/i.test(event.message)
        ? { label: 'Almost done' }
        : { label: event.message }

    default:
      return { label: 'Working' }
  }
}

export function currentPhase(events) {
  const last = events.length ? events[events.length - 1] : null
  return phaseAfter(last)
}

// The phases already finished, for the trail above the live line.
//
// Model calls are included now, and they are the ones worth showing: they are where
// the time goes, so "Planning the analysis — 54s" is the line that explains a wait.
// QUEUED and CLAIMED stay out — nobody watching their question being answered needs to
// know that a worker leased a database row.
export function completedPhases(events) {
  const done = []

  for (const event of events) {
    if (event.kind === 'TOOL_RESULT') {
      const call = [...events].reverse().find((e) => e.kind === 'TOOL_CALL' && e.id < event.id)
      const tool = call?.payload?.tool || event.payload?.tool
      done.push({
        id: event.id,
        label: TOOL_LABELS[tool] || tool || 'Ran a tool',
        detail: event.message,
        ok: event.payload?.ok !== false,
      })
    } else if (event.kind === 'MODEL_CALL') {
      const seconds = event.payload?.seconds
      const writing = /writing the answer/i.test(event.payload?.phase || event.message)
      done.push({
        id: event.id,
        label: writing ? 'Wrote the answer' : 'Planned the analysis',
        detail: seconds ? `${Math.round(seconds)}s of model time` : null,
        ok: true,
      })
    }
  }

  return done
}
