import { useEffect, useState } from 'react'
import { formatDuration } from '../format'
import { completedPhases, currentPhase } from '../phases'

// The glyph cycles rather than spins, so the line reads as "alive" even in a still
// screenshot, and so a reader whose eye is on the text still catches motion in the
// periphery.
const GLYPHS = ['✳', '✻', '✼', '✽']

// The point at which a person stops waiting and starts wondering whether it is broken.
// Measured against the real thing: the schema fetch and the first model turn together
// run 40-70 seconds, so 45 lands inside the first silence rather than after it.
const REASSURE_AFTER_S = 45

// What the user watches for two minutes.
//
// Everything here answers one question: "is anything actually happening?" The activity
// is named, the elapsed time moves every second, the step counter shows progress
// through a bounded budget, and finished work stays on screen with ticks. After
// forty-five seconds it also says, in words, that this is normal — because the honest
// answer to "why is this slow" is "it is a language model on your laptop", and a user
// told that waits happily while a user left guessing reloads the page.
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
        <span className="live-elapsed">
          · {formatDuration(elapsed)}
          {phase.step && phase.maxSteps ? ` · step ${phase.step} of ${phase.maxSteps}` : ''}
        </span>
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
