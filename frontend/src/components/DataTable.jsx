import { formatCell, isNumeric, prettyColumn } from '../format'

// The evidence. Every answer sits above one of these, and the whole premise of the
// system is that a claim is worth what its evidence is worth — so this table has to be
// readable enough that someone actually checks it.
//
// Three decisions do most of that work:
//
//   numbers right-aligned with tabular figures   a column of figures you can scan
//   the header sticky                            row 40 still says what it is
//   a height cap with its own scroll             50 rows never push the answer away
//
// Numeric alignment is decided PER COLUMN, from the data, rather than per cell. A
// column with one null in it is still a column of numbers, and letting that one cell
// left-align would break the ragged-right edge that makes the rest scannable.
export default function DataTable({ table }) {
  if (!table?.rows?.length) return null

  const { columns, rows } = table

  // A ONE-CELL RESULT IS NOT A TABLE.
  //
  // "Is there a relationship between quantity and unit price?" produced a full bordered
  // table, with a sticky header, a column title and a row-count caption, to hold the
  // single value -0.0012 — directly beneath a sentence that had already said -0.001.
  // All the furniture of a table and none of its purpose: nothing to scan, nothing to
  // compare, nothing to sort.
  //
  // The evidence still has to be shown, because the whole premise is that an answer is
  // worth what its evidence is worth. So it is shown as what it actually is: one
  // computed figure, labelled.
  if (rows.length === 1 && columns.length === 1) {
    return (
      <div className="metric">
        <span className="metric-value">{formatCell(rows[0][0])}</span>
        <span className="metric-label">{prettyColumn(columns[0])}</span>
      </div>
    )
  }
  const numericColumn = columns.map((_, index) =>
    rows.some((row) => isNumeric(row[index])) &&
    rows.every((row) => row[index] === null || isNumeric(row[index])),
  )

  return (
    <div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((column, index) => (
                <th key={column} className={numericColumn[index] ? 'is-number' : undefined}>
                  {prettyColumn(column)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, rowIndex) => (
              // eslint-disable-next-line react/no-array-index-key -- rows have no id
              <tr key={rowIndex}>
                {row.map((cell, index) => (
                  <td
                    key={index}
                    className={
                      [
                        numericColumn[index] ? 'is-number' : '',
                        cell === null || cell === undefined ? 'is-null' : '',
                      ]
                        .filter(Boolean)
                        .join(' ') || undefined
                    }
                    title={cell === null ? undefined : String(cell)}
                  >
                    {formatCell(cell)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="table-note">
        {rows.length} row{rows.length === 1 ? '' : 's'} — computed by the database, not
        written by the model
      </p>
    </div>
  )
}
