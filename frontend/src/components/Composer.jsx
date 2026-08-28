import { useEffect, useRef, useState } from 'react'

// The primary action, in the place a chat application puts it.
//
// Enter sends and Shift+Enter adds a line, which is the convention every user of this
// kind of interface already has in their fingers. The textarea grows with its content
// up to a cap rather than scrolling at one line, because a question worth asking a
// data analyst is often two lines long.
//
// It is DISABLED while an analysis runs, and says why. There is one worker: a second
// question would sit PENDING behind the first for two minutes and look like a hang.
// Refusing it with a reason is better than accepting it and appearing broken.
export default function Composer({ dataset, busy, onAsk }) {
  const [text, setText] = useState('')
  const box = useRef(null)

  // Grow to fit, up to the CSS max-height. Reset to auto first or the box can only
  // ever get taller, never shorter, as the text is deleted.
  useEffect(() => {
    const el = box.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${el.scrollHeight}px`
  }, [text])

  function submit() {
    const question = text.trim()
    if (!question || !dataset || busy) return
    onAsk(question)
    setText('')
  }

  const disabled = !dataset || busy

  return (
    <div className="composer">
      <div className="composer-inner">
        <div className="composer-box">
          <textarea
            ref={box}
            rows={1}
            value={text}
            disabled={!dataset}
            placeholder={
              dataset ? `Ask about ${dataset.name}…` : 'Upload a dataset to get started'
            }
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
          />
          <button
            type="button"
            className="send"
            onClick={submit}
            disabled={disabled || !text.trim()}
            title={busy ? 'Wait for the current question to finish' : 'Ask'}
          >
            {busy ? <span className="spin">◠</span> : '↑'}
          </button>
        </div>

        <div className="composer-hint">
          <span>
            {busy
              ? 'Working on your question — one at a time, on one local worker.'
              : 'Every number is computed by the database, never recalled by the model.'}
          </span>
          <span>
            <kbd>Enter</kbd> to send · <kbd>Shift</kbd>+<kbd>Enter</kbd> for a new line
          </span>
        </div>
      </div>
    </div>
  )
}
