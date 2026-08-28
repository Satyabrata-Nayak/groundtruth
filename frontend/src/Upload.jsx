import { useState } from 'react'
import { api } from './api'

// Component 1 of 5. Pick a file, name it, upload it.
export default function Upload({ onUploaded }) {
  const [file, setFile] = useState(null)
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)

  async function submit(event) {
    event.preventDefault()
    if (!file) return
    setBusy(true)
    setError(null)
    try {
      const dataset = await api.uploadDataset(file, name || null)
      setFile(null)
      setName('')
      // Reset the file input by hand: it is an uncontrolled element, so clearing
      // React state does not clear what the browser is showing.
      event.target.reset()
      onUploaded(dataset)
    } catch (err) {
      setError(err.message)
    } finally {
      // In `finally`, so a failed upload re-enables the button. Putting this after
      // the happy path is how a form ends up permanently disabled after one error.
      setBusy(false)
    }
  }

  return (
    <section>
      <h2>1. Upload</h2>
      <form onSubmit={submit}>
        <input
          type="file"
          accept=".csv,.parquet"
          onChange={(e) => setFile(e.target.files[0] ?? null)}
        />
        <input
          type="text"
          placeholder="name (optional)"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <button type="submit" disabled={!file || busy}>
          {busy ? 'uploading…' : 'upload'}
        </button>
      </form>
      {error && <p role="alert">upload failed: {error}</p>}
    </section>
  )
}
