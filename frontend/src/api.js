// Every call to the backend goes through here.
//
// One module rather than fetch() scattered through five components, for one concrete
// reason: error handling. FastAPI reports failures as {"detail": "..."} with a non-2xx
// status, and `fetch` does NOT throw on 4xx or 5xx — it resolves happily with ok=false.
// A component that forgets to check `res.ok` renders `undefined` and looks like a
// frontend bug when the backend said exactly what was wrong. Checking once, here,
// means every caller gets a real Error carrying the server's own message.

async function request(path, options = {}) {
  const res = await fetch(path, options)

  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      // 422 from pydantic is a list of field errors, not a string.
      if (typeof body.detail === 'string') detail = body.detail
      else if (Array.isArray(body.detail)) detail = body.detail.map((e) => e.msg).join('; ')
    } catch {
      // no JSON body — keep the status line
    }
    throw new Error(detail)
  }

  if (res.status === 204) return null

  // A JSON API that answers with HTML means the dev server served its SPA fallback,
  // which means this path is not in vite.config.js's proxy list. Without this check the
  // symptom is `res.json()` throwing "Unexpected token '<'" inside somebody's .catch(),
  // and the visible result is a component rendering nothing at all — which is how the
  // model picker shipped invisible and took a screenshot to notice.
  const type = res.headers.get('content-type') || ''
  if (!type.includes('json')) {
    throw new Error(
      `${path} returned ${type || 'no content-type'} instead of JSON. ` +
        `If this is an API route, add it to the proxy list in vite.config.js.`,
    )
  }

  return res.json()
}

export const api = {
  health: () => request('/healthz'),

  listDatasets: () => request('/datasets'),

  listModels: () => request('/models'),

  getProfile: (datasetId, version) =>
    request(`/datasets/${datasetId}/profile${version ? `?version=${version}` : ''}`),

  deleteDataset: (datasetId) => request(`/datasets/${datasetId}`, { method: 'DELETE' }),

  uploadDataset: (file, name) => {
    const form = new FormData()
    form.append('file', file)
    if (name) form.append('name', name)
    // No Content-Type header set on purpose: the browser has to add it itself so it
    // can include the multipart boundary. Setting it by hand omits the boundary and
    // the server rejects the body as malformed.
    return request('/datasets', { method: 'POST', body: form })
  },

  // `model` and `thinking` are the asker's choice, pinned onto the row so the answer
  // stays explicable later. Both null means "use whatever the worker is configured
  // with", which is not the same as choosing the default explicitly.
  createAnalysis: (datasetId, question, { version, model, thinking, conversationId } = {}) =>
    request('/analyses', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset_id: datasetId,
        question,
        version: version ?? null,
        model: model ?? null,
        thinking: thinking ?? null,
        // Omitted on the first question: the API starts a thread and returns its
        // id, so beginning a conversation never costs an extra round trip.
        conversation_id: conversationId ?? null,
      }),
    }),

  getAnalysis: (id) => request(`/analyses/${id}`),

  // `after` is the id of the last event already seen, so each poll transfers only
  // what is new. Returns {events, next_after, status} — status comes back with the
  // events so a polling client needs one request per tick, not two.
  getEvents: (id, after = 0) => request(`/analyses/${id}/events?after=${after}`),

  cancelAnalysis: (id) => request(`/analyses/${id}/cancel`, { method: 'POST' }),
}
