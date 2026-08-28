import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'

// StrictMode double-invokes effects in development on purpose. That is not a nuisance
// to be switched off: it is what proves the polling effect in Result.jsx cleans up
// after itself. An effect that leaks a timer works fine without StrictMode and leaks
// one poll per mount forever with it.
createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
