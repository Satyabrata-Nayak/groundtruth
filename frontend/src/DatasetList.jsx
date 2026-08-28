import { api } from './api'

// Component 2 of 5. Every dataset, newest first; click one to select it.
export default function DatasetList({ datasets, selectedId, onSelect, onDeleted }) {
  async function remove(id, event) {
    event.stopPropagation()
    if (!confirm('Delete this dataset and all its versions?')) return
    await api.deleteDataset(id)
    onDeleted(id)
  }

  if (datasets.length === 0) {
    return (
      <section>
        <h2>2. Datasets</h2>
        <p>none yet — upload one above.</p>
      </section>
    )
  }

  return (
    <section>
      <h2>2. Datasets</h2>
      <ul>
        {datasets.map((d) => {
          const latest = d.versions.at(-1)
          return (
            <li key={d.id}>
              <label>
                <input
                  type="radio"
                  name="dataset"
                  checked={selectedId === d.id}
                  onChange={() => onSelect(d)}
                />
                <b>{d.name}</b>{' '}
                {latest
                  ? `— v${latest.version}, ${latest.row_count.toLocaleString()} rows × ${latest.column_count} cols`
                  : '— no versions'}
              </label>{' '}
              <button onClick={(e) => remove(d.id, e)}>delete</button>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
