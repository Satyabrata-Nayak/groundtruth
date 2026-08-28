// Turning the event log into "what is happening right now".
//
// THE INSIGHT THIS FILE IS BUILT ON
// --------------------------------
// Events record what ALREADY HAPPENED. `MODEL_CALL` is written after the call returns,
// which means that during the 157 seconds the model is actually thinking, NO EVENT
// ARRIVES AT ALL. A UI that renders the event list is therefore silent for exactly the
// stretch where the user most needs to know something is happening — and silence in
// front of a spinner-less page is what makes people reload and lose their place.
//
// So the current phase is derived from the state AFTER the last event, not from the
// last event itself. The most important row in the table below is `TOOL_RESULT`: a
// tool has just finished, which means a model call is in flight, which means the right
// thing to say is "Thinking about how to answer" — the one message the event log
// cannot give you, because nothing has happened yet to write down.

const TOOL_LABELS = {
  inspect_schema: 'Reading the schema',
  profile_column: 'Profiling a column',
  execute_sql: 'Running a query',
  compare_groups: 'Comparing groups',
  correlation: 'Measuring a correlation',
  create_chart: 'Building a chart',
}

// What each finished event means for the phase that is running NOW.
function phaseAfter(event) {
  if (!event) return { label: 'Starting', detail: null }

  const payload = event.payload || {}

  switch (event.kind) {
    case 'QUEUED':
      return { label: 'Waiting for a worker', detail: null }

    case 'CLAIMED':
    case 'RECLAIMED':
      return { label: 'Reading your data', detail: null }

    // A tool has been announced but has not reported back: it is running.
    case 'TOOL_CALL':
      return {
        label: TOOL_LABELS[payload.tool] || `Running ${payload.tool || 'a tool'}`,
        detail: payload.arguments?.sql ? 'querying DuckDB' : null,
      }

    // THE IMPORTANT ONE. A tool just finished, so the worker has gone back to the
    // model, and that is the long silent stretch.
    case 'TOOL_RESULT':
      return { label: 'Thinking about how to answer', detail: event.message }

    // The model decided something. Either it asked for a tool (which is about to run)
    // or it is writing prose.
    case 'MODEL_CALL':
      return /writing the answer/i.test(event.message)
        ? { label: 'Writing the answer', detail: null }
        : { label: 'Deciding what to compute', detail: null }

    case 'NOTE':
      return /answer written/i.test(event.message)
        ? { label: 'Checking the figures', detail: null }
        : { label: event.message, detail: null }

    default:
      return { label: 'Working', detail: null }
  }
}

// The phase currently running, plus which step of the budget it belongs to.
export function currentPhase(events) {
  const last = events.length ? events[events.length - 1] : null
  const phase = phaseAfter(last)

  // `step` is only carried by MODEL_CALL, so the newest one is the current step.
  let step = null
  let maxSteps = null
  for (let i = events.length - 1; i >= 0; i -= 1) {
    const found = events[i]
    if (found.kind === 'MODEL_CALL' && found.payload?.step) {
      step = found.payload.step
      const match = /step\s+\d+\s*\/\s*(\d+)/.exec(found.message || '')
      maxSteps = match ? Number(match[1]) : null
      break
    }
  }

  return { ...phase, step, maxSteps }
}

// The phases that are already finished, for the trail above the live line. Only real
// work is listed: QUEUED and CLAIMED are plumbing, and a user watching their question
// being answered does not need to know a worker leased a row.
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
    }
  }
  return done
}
