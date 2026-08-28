import { useEffect, useState } from 'react'
import { api } from '../api'

// The blank page problem, solved with the user's own column names.
//
// "Ask a question about your data" is a prompt that stalls people, because the honest
// first reaction to it is "which questions can this thing actually answer?". Three
// clickable examples built from the dataset's real schema answer that instantly, and
// they double as a demonstration that the app has already read the file.
//
// The suggestions are generated rather than hardcoded, so they name Country and
// Quantity on the retail data and region and revenue on a sales extract. A hardcoded
// example that mentions a column the user does not have teaches them nothing and looks
// like a template.
export default function Empty({ dataset, onPick }) {
  const [suggestions, setSuggestions] = useState([])

  useEffect(() => {
    if (!dataset) {
      setSuggestions([])
      return undefined
    }
    let cancelled = false
    api
      .getProfile(dataset.id)
      .then((profile) => !cancelled && setSuggestions(buildSuggestions(profile)))
      .catch(() => !cancelled && setSuggestions([]))
    return () => {
      cancelled = true
    }
  }, [dataset])

  if (!dataset) {
    return (
      <div className="empty">
        <div className="empty-title">Nothing loaded yet</div>
        <p className="empty-sub">
          Drop a CSV into the sidebar. It is profiled exactly — no sampling, no estimates
          — and stays on this machine.
        </p>
      </div>
    )
  }

  return (
    <div className="empty">
      <div className="empty-title">Ask {dataset.name} something</div>
      <p className="empty-sub">
        A local model decides what to compute. The database does the computing, and every
        figure in the answer is traced back to it.
      </p>
      <div className="suggestions">
        {suggestions.map((question) => (
          <button type="button" className="suggestion" key={question} onClick={() => onPick(question)}>
            {question}
          </button>
        ))}
      </div>
    </div>
  )
}

// Picks columns the way the fixed engine does, and for the same reasons: a grouping
// column has to have few enough distinct values to be worth grouping by, and a metric
// must not be an identifier wearing a number's clothes.
function buildSuggestions(profile) {
  const columns = profile?.columns ?? []

  const groupable = columns.filter(
    (c) =>
      c.semantic_type === 'categorical' &&
      !c.is_constant &&
      !c.is_high_cardinality &&
      (c.distinct_count ?? 0) > 1 &&
      (c.distinct_count ?? 0) <= 60,
  )
  const metrics = columns.filter(
    (c) => c.semantic_type === 'numeric' && !c.is_constant && !c.is_high_cardinality,
  )

  const out = []
  if (groupable[0] && metrics[0]) {
    out.push(`Which ${groupable[0].name} has the highest total ${metrics[0].name}?`)
  }
  if (metrics.length >= 2) {
    out.push(`What is the relationship between ${metrics[0].name} and ${metrics[1].name}?`)
  }
  if (groupable[0]) {
    out.push(`How many rows are there per ${groupable[0].name}?`)
  }
  if (out.length < 3) out.push('What does this dataset contain, and how complete is it?')
  return out.slice(0, 3)
}
