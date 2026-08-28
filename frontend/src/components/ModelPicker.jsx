import { useEffect, useRef, useState } from 'react'

// Choosing which model answers, the way a chat app does it — a small control by the
// composer that opens a menu of options.
//
// WHY THE MENU CARRIES NUMBERS AND NOT ADJECTIVES
// -----------------------------------------------
// "Fast" and "smart" are what every model picker says and they help nobody decide. The
// real question is "will this answer MY question correctly, and how long will I wait",
// and both halves were measured on this project's own evaluation set:
//
//     Qwen3 4B              60% correct    1-3 minutes
//     Qwen2.5 3B            29% correct    ~3 seconds
//
// Fifty times faster for half the accuracy is a genuine trade with no right answer,
// which is exactly the kind of decision that belongs to the person asking rather than
// to a default nobody can see.
//
// Each option also lists what it is BAD at. A chooser that only lists strengths is an
// advert, and the user is picking between two things that are each bad at something.
export default function ModelPicker({ models, value, thinking, onModel, onThinking }) {
  const [open, setOpen] = useState(false)
  const root = useRef(null)

  // Click-away and Escape. Without both, a menu opened by accident has to be dismissed
  // by choosing something from it, which is how people end up on the wrong model.
  useEffect(() => {
    if (!open) return undefined
    function onDown(event) {
      if (!root.current?.contains(event.target)) setOpen(false)
    }
    function onKey(event) {
      if (event.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDown)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])

  if (!models?.length) return null

  const selected = models.find((m) => m.name === value) ?? models.find((m) => m.is_default)
  if (!selected) return null

  return (
    <div className="picker" ref={root}>
      <button
        type="button"
        className="picker-trigger"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span>{selected.label}</span>
        <span className="picker-speed">{selected.speed}</span>
        <span className="picker-caret" aria-hidden="true">
          ⌄
        </span>
      </button>

      {/* ALWAYS RENDERED, disabled when it cannot apply.

          The first version hid this entirely for a model with no reasoning step, on the
          reasoning that a control which does nothing should not be offered. That was
          wrong in the way that matters: selecting Qwen2.5 made the button vanish, the
          choice persisted to localStorage, and from the outside "the reasoning button
          is missing" is indistinguishable from "the reasoning button is broken".

          A disabled control that says why is a fact about the model. An absent one is a
          bug report waiting to be filed. */}
      <button
        type="button"
        className={`toggle ${!selected.reasons ? 'is-na' : thinking === false ? '' : 'is-on'}`}
        disabled={!selected.reasons}
        onClick={() => onThinking(thinking === false ? null : false)}
        title={
          !selected.reasons
            ? `${selected.label} has no reasoning step to switch off — it answers directly. `
              + `Choose Qwen3 4B to control this.`
            : thinking === false
              ? 'Reasoning off. Measured: this does NOT make it faster (42.1s vs 43.6s), '
                + 'and the model’s scratch work can end up inside the answer.'
              : 'Reasoning on. The model works through the question before answering.'
        }
      >
        <span className="toggle-track">
          <span className="toggle-knob" />
        </span>
        reasoning {!selected.reasons ? 'n/a' : thinking === false ? 'off' : 'on'}
      </button>

      {open && (
        <div className="picker-menu" role="listbox">
          {models.map((model) => (
            <button
              type="button"
              key={model.name}
              role="option"
              aria-selected={model.name === selected.name}
              className={`picker-option ${model.name === selected.name ? 'is-selected' : ''}`}
              disabled={!model.available}
              onClick={() => {
                onModel(model.name)
                setOpen(false)
              }}
            >
              <div className="picker-option-head">
                <span className="picker-option-name">{model.label}</span>
                {/* Where it runs is the first thing to know: one of these leaves the
                    machine and one does not. */}
                <span className={`badge ${model.provider === 'groq' ? 'is-hosted' : 'is-quiet'}`}>
                  {model.provider === 'groq' ? 'hosted' : 'local'}
                </span>
                {model.accuracy_pct !== null && (
                  <span className="badge">{model.accuracy_pct}% correct</span>
                )}
                <span className="badge is-quiet">{model.speed}</span>
                <span className="badge is-quiet">{model.cost}</span>
                {model.preview && <span className="badge is-warn">preview</span>}
              </div>

              <p className="picker-option-tagline">{model.tagline}</p>

              <ul className="picker-points">
                {model.good_at.map((point) => (
                  <li key={point}>
                    <span className="point-mark is-good">+</span>
                    {point}
                  </li>
                ))}
                {model.weak_at.map((point) => (
                  <li key={point}>
                    <span className="point-mark is-bad">−</span>
                    {point}
                  </li>
                ))}
              </ul>

              {/* Why it cannot be used, in the terms of ITS provider. "not installed"
                  is meaningless for a hosted model and "no API key" is meaningless for
                  a local one. */}
              {!model.available && (
                <p className="picker-missing">
                  {model.provider === 'groq' ? (
                    <>
                      no API key — put <code>GROQ_API_KEY</code> in <code>.env</code> and
                      restart the API and worker
                    </>
                  ) : (
                    <>
                      not installed — run <code>ollama pull {model.name}</code>
                    </>
                  )}
                </p>
              )}
            </button>
          ))}

          <p className="picker-foot">
            Percentages are this project&rsquo;s own evaluation set: {' '}
            <code>python -m eval.runner --agent local-model</code>. Hosted models send
            your tool <em>results</em> — never your file — to Groq; local models send
            nothing anywhere.
          </p>
        </div>
      )}
    </div>
  )
}
