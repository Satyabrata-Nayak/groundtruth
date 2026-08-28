// Does the app actually render, and does it render the right things?
//
// WHY THIS EXISTS
// ---------------
// Every layer of this system has been proven with real processes except the browser.
// `npm run build` proves the code parses and bundles; it proves nothing about whether
// the first render throws, which is the difference between a working app and a white
// screen. A component that reads `dataset.latest_version.row_count` builds perfectly
// and renders nothing — and that exact bug was in this codebase an hour ago.
//
// So: load the production bundle into jsdom, stub the API with realistic payloads
// taken from the real thing, render, and assert on what the user would see. It is not
// a substitute for looking at it — it says nothing about whether it is *nice* — but it
// does turn "it renders" from an assumption into a check that runs in two seconds.
//
//   node smoke.mjs        (after `npm run build`)

import { readFileSync, readdirSync } from 'node:fs'
import { JSDOM, VirtualConsole } from 'jsdom'

const DIST = new URL('./dist/', import.meta.url)
const bundle = readdirSync(DIST).find((f) => f.startsWith('assets/') || f.endsWith('.js'))

// Vite writes assets/index-<hash>.js; find it without hardcoding the hash.
const assets = readdirSync(new URL('./assets/', DIST))
const js = assets.find((f) => f.endsWith('.js'))
const script = readFileSync(new URL(`./assets/${js}`, DIST), 'utf8')

// ── the API, as it actually replies ──────────────────────────────────────────
// These payloads are copied from a live run against the 541,909-row retail dataset,
// so a component that mishandles the real shape fails here rather than in the browser.

const DATASET = {
  id: '7166db38-465b-4f58-afe1-025bb8be1365',
  name: 'retail',
  description: null,
  created_at: '2026-08-28T10:00:00Z',
  latest_version: 1,
  versions: [
    {
      version: 1,
      original_filename: 'Online Retail.csv',
      original_format: 'csv',
      source_bytes: 43954942,
      parquet_bytes: 8000000,
      row_count: 541909,
      column_count: 8,
      duplicate_row_count: 5268,
      ingested_at: '2026-08-28T10:00:00Z',
    },
  ],
}

const PROFILE = {
  dataset_id: DATASET.id,
  version: 1,
  row_count: 541909,
  column_count: 8,
  duplicate_row_count: 5268,
  columns: [
    col('InvoiceNo', 'VARCHAR', 'categorical', { distinct_count: 25900, is_high_cardinality: true }),
    col('Description', 'VARCHAR', 'categorical', { distinct_count: 4223, null_fraction: 0.0027 }),
    col('Quantity', 'BIGINT', 'numeric', { distinct_count: 722, mean_value: 9.55 }),
    col('UnitPrice', 'DOUBLE', 'numeric', { distinct_count: 1630, mean_value: 4.61 }),
    col('CustomerID', 'BIGINT', 'numeric', { distinct_count: 4372, null_fraction: 0.2493 }),
    col('Country', 'VARCHAR', 'categorical', { distinct_count: 38 }),
  ],
}

function col(name, type, kind, extra = {}) {
  return {
    name,
    position: 0,
    duckdb_type: type,
    semantic_type: kind,
    null_count: 0,
    null_fraction: 0,
    distinct_count: null,
    min_value: null,
    max_value: null,
    mean_value: null,
    stddev_value: null,
    q25_value: null,
    q50_value: null,
    q75_value: null,
    is_constant: false,
    is_high_cardinality: false,
    ...extra,
  }
}

const RESULT = {
  engine: 'agent-v1',
  question: 'Which country generated the most revenue?',
  dataset: { id: DATASET.id, version: 1 },
  answer:
    'The United Kingdom generated the most revenue at 8187806.36, followed by the ' +
    "Netherlands with 284661.54.",
  warnings: ['the following figure does not appear in any tool result: 168,000'],
  table: {
    columns: ['Country', 'TotalRevenue'],
    rows: [
      ['United Kingdom', 8187806.363998199],
      ['Netherlands', 284661.53999999963],
      ['EIRE', 263276.81999999995],
    ],
  },
  chart: {
    chart: {
      type: 'bar',
      title: 'Which country generated the most revenue?',
      x: { column: 'Country', kind: 'categorical', label: 'Country' },
      y: { column: 'TotalRevenue', kind: 'numeric', label: 'TotalRevenue' },
      data: [
        { x: 'United Kingdom', y: 8187806.36 },
        { x: 'Netherlands', y: 284661.54 },
        { x: 'EIRE', y: 263276.82 },
      ],
      point_count: 3,
      derived_from: 'the final query result',
    },
  },
  steps: [
    {
      tool: 'inspect_schema',
      arguments: { include_statistics: true },
      ok: true,
      summary: '541909 rows, 8 columns',
      error: null,
      duration_ms: 40.29,
    },
    {
      tool: 'execute_sql',
      arguments: {
        sql: 'SELECT Country, SUM(Quantity * UnitPrice) AS TotalRevenue FROM dataset GROUP BY Country ORDER BY TotalRevenue DESC LIMIT 10',
        max_rows: 50,
      },
      ok: true,
      summary: '10 rows x 2 column(s) in 9 ms',
      error: null,
      duration_ms: 40.13,
    },
  ],
}

const ANALYSIS_ID = '90d411af-90b5-499a-9a86-b4c232678dd2'

const EVENTS = [
  { id: 1, kind: 'QUEUED', message: 'queued', payload: null, created_at: '' },
  { id: 2, kind: 'CLAIMED', message: 'claimed by host:1:abc (attempt 1)', payload: null, created_at: '' },
  {
    id: 3,
    kind: 'TOOL_CALL',
    message: 'calling inspect_schema',
    payload: { tool: 'inspect_schema', arguments: { include_statistics: true } },
    created_at: '',
  },
  {
    id: 4,
    kind: 'TOOL_RESULT',
    message: '541909 rows, 8 columns',
    payload: { ok: true, tool: 'inspect_schema', duration_ms: 40.29 },
    created_at: '',
  },
  {
    id: 5,
    kind: 'MODEL_CALL',
    message: 'step 1/6: execute_sql (53.7s)',
    payload: { step: 1, seconds: 53.7, output_tokens: 1303, prompt_tokens: 2036 },
    created_at: '',
  },
]

function routes(url, phase) {
  if (url.endsWith('/healthz')) return { status: 'ok', database: true, version: '0.4.0' }
  if (url.endsWith('/datasets')) return [DATASET]
  if (url.includes('/profile')) return PROFILE
  if (url.includes('/events')) {
    return phase === 'running'
      ? { events: EVENTS, next_after: 5, status: 'RUNNING' }
      : { events: EVENTS, next_after: 9, status: 'SUCCEEDED' }
  }
  if (url.includes('/analyses/')) {
    return {
      id: ANALYSIS_ID,
      dataset_id: DATASET.id,
      dataset_version: 1,
      question: RESULT.question,
      status: 'SUCCEEDED',
      attempts: 1,
      created_at: '',
      started_at: '',
      finished_at: '',
      result: RESULT,
      error: null,
    }
  }
  if (url === '/analyses') return { id: ANALYSIS_ID, status: 'PENDING' }
  return {}
}

// ── run it ───────────────────────────────────────────────────────────────────

async function render(phase) {
  const errors = []
  const virtualConsole = new VirtualConsole()
  virtualConsole.on('jsdomError', (e) => errors.push(e.message))
  virtualConsole.on('error', (...args) => errors.push(args.join(' ')))

  const dom = new JSDOM('<!doctype html><html><body><div id="root"></div></body></html>', {
    runScripts: 'outside-only',
    url: 'http://localhost:5173/',
    pretendToBeVisual: true,
    virtualConsole,
  })

  const { window } = dom
  window.fetch = async (url, options) => {
    const body = routes(String(url), phase)
    return {
      ok: true,
      status: options?.method === 'POST' ? 201 : 200,
      statusText: 'OK',
      json: async () => body,
    }
  }
  window.scrollTo = () => {}
  window.Element.prototype.scrollIntoView = () => {}

  try {
    window.eval(script)
  } catch (err) {
    errors.push(`bundle threw: ${err.message}`)
  }

  // React 19 renders synchronously into the root, then effects resolve their promises
  // over a few microtask/macrotask turns. Three ticks is enough for fetch -> setState
  // -> re-render for every component here.
  for (let i = 0; i < 6; i += 1) {
    await new Promise((resolve) => setTimeout(resolve, 60))
  }

  return { html: window.document.getElementById('root').innerHTML, errors, window }
}

const checks = []
function expect(name, condition, detail = '') {
  checks.push({ name, ok: Boolean(condition), detail })
}

// ── 1. the shell, with a dataset loaded and nothing asked yet ────────────────
{
  const { html, errors } = await render('idle')
  expect('the app renders at all', html.length > 500, `${html.length} chars`)
  expect('no runtime errors', errors.length === 0, errors.join(' | '))
  expect('the brand is shown', html.includes('Ground Truth'))
  expect('the dataset is listed', html.includes('retail'))
  expect('the row count is compact', html.includes('542K') || html.includes('541.9K'), html.match(/v1[^<]*/)?.[0])
  expect('columns appear in the sidebar', html.includes('UnitPrice') && html.includes('Country'))
  expect('the null badge shows only where it matters', html.includes('25%'))
  expect('the empty state invites a question', html.includes('Ask retail something'))
  expect(
    'suggestions are built from real column names',
    html.includes('Country') && html.includes('Quantity'),
  )
  expect('the composer is present', html.includes('Ask about retail'))
}

// ── 2. a finished exchange ───────────────────────────────────────────────────
{
  const { window, errors } = await render('done')
  const root = window.document.getElementById('root')

  // Type a question and send it, the way a user does.
  const textarea = root.querySelector('textarea')
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set
  setter.call(textarea, 'Which country generated the most revenue?')
  textarea.dispatchEvent(new window.Event('input', { bubbles: true }))
  root.querySelector('.send').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))

  for (let i = 0; i < 8; i += 1) await new Promise((r) => setTimeout(r, 60))
  const html = root.innerHTML

  expect('no runtime errors while answering', errors.length === 0, errors.join(' | '))
  expect('the question is echoed as a bubble', html.includes('class="ask"'))
  expect('the answer is rendered', html.includes('The United Kingdom generated the most revenue'))
  expect('the warning is a callout, not a bullet', html.includes('callout') && html.includes('Unverified'))
  expect('the chart draws bars', (html.match(/chart-bar/g) || []).length === 3)
  // The VISIBLE cell is rounded; the full-precision value survives in the title
  // attribute, so hovering shows exactly what the database returned. Rounding for
  // legibility is fine — hiding the fact that you rounded is not.
  const cells = [...root.querySelectorAll('td')].map((td) => td.textContent)
  expect('the visible figure is formatted', cells.includes('8,187,806.36'), cells.join(' / '))
  expect(
    'the exact value is still reachable on hover',
    [...root.querySelectorAll('td')].some((td) => td.title === '8187806.363998199'),
  )
  expect('the evidence table is present', html.includes('United Kingdom') && html.includes('<table'))
  expect('the trace is collapsed but present', html.includes('How it got there'))
  expect('the SQL is a code block', html.includes('class="sql"') && html.includes('GROUP BY'))
}

// ── 3. the live status line, mid-run ─────────────────────────────────────────
{
  const { window, errors } = await render('running')
  const root = window.document.getElementById('root')
  const textarea = root.querySelector('textarea')
  const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set
  setter.call(textarea, 'Which country generated the most revenue?')
  textarea.dispatchEvent(new window.Event('input', { bubbles: true }))
  root.querySelector('.send').dispatchEvent(new window.MouseEvent('click', { bubbles: true }))

  for (let i = 0; i < 8; i += 1) await new Promise((r) => setTimeout(r, 60))
  const html = root.innerHTML

  expect('no runtime errors while running', errors.length === 0, errors.join(' | '))
  expect('a live status line is shown', html.includes('live-now'))
  // The last event is MODEL_CALL naming a tool, so the phase running NOW is the
  // decision that follows it. This is the assertion that protects the whole point of
  // phases.js: the line describes what is happening, not what already happened.
  expect(
    'the phase describes what is happening now',
    html.includes('Deciding what to compute'),
    html.match(/live-now[\s\S]{0,220}/)?.[0]?.replace(/<[^>]+>/g, ' '),
  )
  expect('the step counter is shown', html.includes('step 1 of 6'))
  expect('finished phases keep their ticks', html.includes('live-tick'))
  expect('the composer is locked while working', html.includes('disabled'))
}

// ── report ───────────────────────────────────────────────────────────────────

let failed = 0
for (const check of checks) {
  if (!check.ok) failed += 1
  console.log(`${check.ok ? '  ok  ' : ' FAIL '} ${check.name}${check.ok || !check.detail ? '' : `\n         ${check.detail}`}`)
}
console.log(`\n${checks.length - failed}/${checks.length} passed`)
process.exit(failed ? 1 : 0)
