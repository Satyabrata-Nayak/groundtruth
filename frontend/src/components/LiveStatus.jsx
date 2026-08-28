import { useEffect, useState } from 'react'
import { formatDuration } from '../format'
import { completedPhases, currentPhase } from '../phases'

// The glyph cycles rather than spins, so the line reads as "alive" even in a still
// screenshot, and so a reader whose eye is on the text still catches motion in the
// periphery.
const GLYPHS = ['✳', '✻', '✼', '✽']

// The point at which a person stops waiting and starts wondering whether it is broken.
// Measured against the real thing: the planning turn alone runs 45-90 seconds, so 30
// lands inside the first silence rather than after it.
const REASSURE_AFTER_S = 30

// What the user watches while the model works.
//
// Everything here answers one question: "is anything actually happening?" The activity
// is named, the elapsed time moves every second, and finished work stays on screen with
// ticks and the time it took. After thirty seconds it also says, in words, that this is
// normal — because the honest answer to "why is this slow" is "it is a language model
// on your laptop", and a user told that waits happily while a user left guessing
// reloads the page.
//
// There is deliberately no "step N of 6" any more. It came from the newest MODEL_CALL,
// which is written when a call FINISHES, so it displayed the step that had just ended
// and sat on "step 1 of 6" for the whole of step 2. A counter that is always one behind
// is worse than no counter.
export default function LiveStatus({ events, elapsed }) {
  const [frame, setFrame] = useState(0)

  useEffect(() => {
    const timer = setInterval(() => setFrame((n) => n + 1), 420)
    return () => clearInterval(timer)
  }, [])

  const phase = currentPhase(events)
  const done = completedPhases(events)

  return (
    <div className="live">
      {done.length > 0 && (
        <div className="live-done">
          {done.map((step) => (
            <div className="live-step" key={step.id}>
              <span className="live-tick">{step.ok ? '✓' : '✕'}</span>
              <span>{step.label}</span>
              {step.detail && <span className="muted">— {step.detail}</span>}
            </div>
          ))}
        </div>
      )}

      <div className="live-now" aria-live="polite">
        <span className="live-glyph" aria-hidden="true">
          {GLYPHS[frame % GLYPHS.length]}
        </span>
        <span>{phase.label}</span>
        <span className="live-elapsed">· {formatDuration(elapsed)}</span>
      </div>

      {elapsed >= REASSURE_AFTER_S && (
        <p className="live-reassure">
          This runs entirely on your machine — no data leaves it. A local model takes
          anywhere up to three minutes to work through a question this size.
        </p>
      )}
    </div>
  )
}
