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

      {/* Shown next to the model rather than inside the menu: it is a property of the
          current choice, and it only applies to a model that reasons at all. */}
      {selected.reasons && (
        <button
          type="button"
          className={`toggle ${thinking === false ? '' : 'is-on'}`}
          onClick={() => onThinking(thinking === false ? null : false)}
          title={
            thinking === false
              ? 'Reasoning off. Measured: this does NOT make it faster, and its scratch work can end up in the answer.'
              : 'Reasoning on. The model works through the question before answering.'
          }
        >
          <span className="toggle-dot" />
          reasoning {thinking === false ? 'off' : 'on'}
        </button>
      )}

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
                {model.accuracy_pct !== null && (
                  <span className="badge">{model.accuracy_pct}% correct</span>
                )}
                <span className="badge is-quiet">{model.speed}</span>
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

              {!model.available && (
                <p className="picker-missing">
                  not installed — run <code>ollama pull {model.name}</code>
                </p>
              )}
            </button>
          ))}

          <p className="picker-foot">
            Percentages are this project&rsquo;s own evaluation set: {' '}
            <code>python -m eval.runner --agent local-model</code>. Both models score 0%
            on &ldquo;why&rdquo; questions.
          </p>
        </div>
      )}
    </div>
  )
}
