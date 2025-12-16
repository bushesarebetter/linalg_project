import { useMemo, useState, useEffect } from 'react'
import './App.css'

function transpose(M) {
  return M[0].map((_, j) => M.map((row) => row[j]))
}

function matMul(A, B) {
  const m = A.length, p = B[0].length, n = A[0].length
  const C = Array.from({ length: m }, () => Array(p).fill(0))
  for (let i = 0; i < m; i++) for (let k = 0; k < n; k++) for (let j = 0; j < p; j++) C[i][j] += A[i][k] * B[k][j]
  return C
}

function matVecMul(A, v) {
  return A.map((row) => row.reduce((s, val, i) => s + val * v[i], 0))
}

function dot(a, b) { return a.reduce((s, x, i) => s + x * b[i], 0) }

function gaussSolve(A, b) {
  // simple Gaussian elimination (in-place copy)
  const n = A.length
  const M = A.map((r) => r.slice())
  const y = b.slice()
  for (let i = 0; i < n; i++) {
    // pivot
    let piv = i
    for (let r = i; r < n; r++) if (Math.abs(M[r][i]) > Math.abs(M[piv][i])) piv = r
    if (Math.abs(M[piv][i]) < 1e-12) return null
    ;[M[i], M[piv]] = [M[piv], M[i]]
    ;[y[i], y[piv]] = [y[piv], y[i]]
    const div = M[i][i]
    for (let j = i; j < n; j++) M[i][j] /= div
    y[i] /= div
    for (let r = 0; r < n; r++) if (r !== i) {
      const factor = M[r][i]
      for (let c = i; c < n; c++) M[r][c] -= factor * M[i][c]
      y[r] -= factor * y[i]
    }
  }
  return y
}

function solveLeastSquares(A, b) {
  const At = transpose(A)
  const AtA = matMul(At, A)
  const Atb = matVecMul(At, b)
  return gaussSolve(AtA, Atb)
}

function makeDesign(xs, deg) {
  return xs.map((x) => {
    const row = []
    for (let p = 0; p <= deg; p++) row.push(Math.pow(x, p))
    return row
  })
}

const problems = [
  {
    title: 'Fit a line y = ax + b',
    desc: 'Find coefficients [a, b] that best fit the points (0,1), (1,2), (2,3).',
    A: makeDesign([0, 1, 2], 1),
    b: [1, 2, 3],
  },
  {
    title: 'Noisy line',
    desc: 'Fit y = ax + b to points (0,1), (1,2), (2,2). Enter matrix A.',
    A: makeDesign([0, 1, 2], 1),
    b: [1, 2, 2],
  },
  {
    title: 'Quadratic fit',
    desc: 'Fit y = ax^2 + bx + c to points (0,1), (1,4), (2,11), (3,22). Enter matrix A.',
    A: makeDesign([0, 1, 2, 3], 2),
    b: [1, 4, 11, 22],
  },
  {
    title: 'Projection coefficients',
    desc: 'Find coefficients that best express b as a combination of the two columns of A.',
    A: makeDesign([0, 1, 2], 1),
    editableA: false,
    b: [2, 2.5, 3],
  },
  {
    title: 'Small system',
    desc: 'Solve the least squares problem for this overdetermined system.',
    A: [[1, 2, 0], [0, 1, 1], [1, 0, 1], [2, 1, 1]],
    editableA: false,
    b: [1, 0, 2, 3],
  },
]

function formatVector(v) { return '[' + v.map((x) => Number(x).toFixed(3)).join(', ') + ']' }

function floatToFraction(x, maxDen = 100) {
  if (!isFinite(x)) return null
  const sign = x < 0 ? -1 : 1
  x = Math.abs(x)
  if (x === 0) return { n: 0, d: 1 }
  // if already very close to integer, return integer
  if (Math.abs(x - Math.round(x)) < 1e-8) return { n: sign * Math.round(x), d: 1 }

  let maxIter = 64
  let h1 = 1, h2 = 0
  let k1 = 0, k2 = 1
  let b = x
  while (maxIter--) {
    const a = Math.floor(b)
    const h = a * h1 + h2
    const k = a * k1 + k2
    if (k > maxDen) break
    h2 = h1; h1 = h
    k2 = k1; k1 = k
    const frac = b - a
    if (Math.abs(frac) < 1e-12) break
    b = 1 / frac
  }
  return { n: sign * h1, d: k1 }
}

function formatCell(x) {
  if (typeof x !== 'number') return String(x)
  if (!isFinite(x)) return String(x)
  // show integer if close
  const rounded = Math.round(x)
  if (Math.abs(x - rounded) < 1e-2) return String(rounded)
  const frac = floatToFraction(x, 100)
  if (frac && frac.d !== 1) return `${frac.n}/${frac.d}`
  // fallback: show 0 decimal places
  return String(Math.round(x))
}

function gcd(a, b) {
  a = Math.abs(a); b = Math.abs(b)
  while (b) { const t = b; b = a % b; a = t }
  return a
}

class Fraction {
  constructor(n, d = 1) {
    if (d === 0) throw new Error('zero denominator')
    if (d < 0) { n = -n; d = -d }
    const g = gcd(n, d)
    this.n = n / g
    this.d = d / g
  }
  static fromNumber(x, maxDen = 1000) {
    if (!isFinite(x)) return null
    if (Math.abs(x - Math.round(x)) < 1e-12) return new Fraction(Math.round(x), 1)
    const f = floatToFraction(x, maxDen)
    return new Fraction(f.n, f.d)
  }
  add(b) { const n = this.n * b.d + b.n * this.d; const d = this.d * b.d; return new Fraction(n, d) }
  sub(b) { const n = this.n * b.d - b.n * this.d; const d = this.d * b.d; return new Fraction(n, d) }
  mul(b) { return new Fraction(this.n * b.n, this.d * b.d) }
  div(b) { return new Fraction(this.n * b.d, this.d * b.n) }
  neg() { return new Fraction(-this.n, this.d) }
  equals(b) { return this.n === b.n && this.d === b.d }
  valueOf() { return this.n / this.d }
  toString() { return this.d === 1 ? String(this.n) : `${this.n}/${this.d}` }
}

function parseToFraction(s) {
  if (s instanceof Fraction) return s
  if (typeof s !== 'string') return null
  const t = s.trim()
  if (t.length === 0) return null
  if (t.includes('/')) {
    const parts = t.split('/')
    if (parts.length !== 2) return null
    const a = parseInt(parts[0].trim(), 10)
    const b = parseInt(parts[1].trim(), 10)
    if (Number.isNaN(a) || Number.isNaN(b) || b === 0) return null
    return new Fraction(a, b)
  }
  const num = parseFloat(t)
  if (Number.isNaN(num)) return null
  return Fraction.fromNumber(num, 1000)
}

function solveLeastSquaresExact(A, b) {
  // A: numeric matrix (arrays of numbers or numeric-strings) -> convert to Fractions
  const Af = A.map((row) => row.map((v) => {
    if (v instanceof Fraction) return v
    if (typeof v === 'string') {
      const pf = parseToFraction(v)
      if (pf) return pf
      const num = parseFloat(v)
      return Fraction.fromNumber(num, 1000)
    }
    if (typeof v === 'number') return Fraction.fromNumber(v, 1000)
    return Fraction.fromNumber(Number(v), 1000)
  }))
  const bf = b.map((v) => typeof v === 'string' ? (parseToFraction(v) || Fraction.fromNumber(parseFloat(v))) : Fraction.fromNumber(v))
  const mt = Af[0].length
  // build AtA
  const AtAf = Array.from({ length: mt }, () => Array(mt).fill(null))
  for (let i = 0; i < mt; i++) for (let j = 0; j < mt; j++) {
    let s = new Fraction(0, 1)
    for (let k = 0; k < Af.length; k++) s = s.add(Af[k][i].mul(Af[k][j]))
    AtAf[i][j] = s
  }
  const Atbf = Array.from({ length: mt }, (_, i) => {
    let s = new Fraction(0, 1)
    for (let k = 0; k < Af.length; k++) s = s.add(Af[k][i].mul(bf[k]))
    return s
  })
  // Solve linear system AtAf * x = Atbf using Gaussian elimination with Fractions
  const N = mt
  const M = AtAf.map((r) => r.map((c) => new Fraction(c.n, c.d)))
  const y = Atbf.map((c) => new Fraction(c.n, c.d))
  for (let i = 0; i < N; i++) {
    // pivot: find row with non-zero at column i
    let piv = i
    for (let r = i; r < N; r++) if (M[r][i].n !== 0) { piv = r; break }
    if (M[piv][i].n === 0) return null
    [M[i], M[piv]] = [M[piv], M[i]]
    [y[i], y[piv]] = [y[piv], y[i]]
    const div = M[i][i]
    for (let j = i; j < N; j++) M[i][j] = M[i][j].div(div)
    y[i] = y[i].div(div)
    for (let r = 0; r < N; r++) if (r !== i) {
      const factor = M[r][i]
      for (let c = i; c < N; c++) M[r][c] = M[r][c].sub(factor.mul(M[i][c]))
      y[r] = y[r].sub(factor.mul(y[i]))
    }
  }
  return y
}

function MatrixView({ matrix, label }) {
  return (
    <div className="matrix-grid">
      <div className="matrix-label">{label}</div>
      <table className="matrix-table">
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              {row.map((cell, j) => {
                const val = typeof cell === 'string' ? (Number.isNaN(Number(cell)) ? cell : Number(cell)) : cell
                return <td key={j}>{formatCell(val)}</td>
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function VectorView({ vector, label }) {
  return (
    <div className="matrix-grid vector">
      <div className="matrix-label">{label}</div>
      <table className="vector-table">
        <tbody>
          {vector.map((v, i) => (
            <tr key={i}><td>{formatCell(v)}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function MatrixEditor({ matrixPlaceholder, matrixValues, onChange, label }) {
  return (
    <div className="matrix-grid">
      <div className="matrix-label">{label}</div>
      <table className="matrix-table">
        <tbody>
          {matrixPlaceholder.map((row, i) => (
            <tr key={i}>
              {row.map((_, j) => (
                <td key={j}>
                              {(() => {
                                const val = (matrixValues[i] && matrixValues[i][j]) || ''
                                const valid = parseToFraction(String(val)) !== null
                                const cls = 'cell-input' + (val === '' || !valid ? ' invalid' : '')
                                return (
                                  <input
                                    className={cls}
                                    value={val}
                                    placeholder={''}
                                    onChange={(e) => onChange(i, j, e.target.value)}
                                  />
                                )
                              })()}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function App() {
  const [index, setIndex] = useState(0)
  const [inputs, setInputs] = useState([])
  const [message, setMessage] = useState('')
  const [userA, setUserA] = useState([])

  const current = problems[index]

  // numeric fallback solution (float) based on current A (for sizing)
  const floatSolution = useMemo(() => {
    try {
      // parse numeric A from current.A (placeholders) if user hasn't filled
      const Afloat = current.A
      return solveLeastSquares(Afloat, current.b)
    } catch { return null }
  }, [index])

  // initialize inputs and blank A values when problem changes
  useEffect(() => {
    const cols = current.A[0].length
    setInputs(Array(cols).fill(''))
    setMessage('')
    if (current.editableA === false) {
      const filled = current.A.map((row) => row.map((cell) => {
        if (cell instanceof Fraction) return cell.toString()
        if (typeof cell === 'number') {
          const fr = Fraction.fromNumber(cell, 1000)
          return fr ? fr.toString() : String(cell)
        }
        return String(cell)
      }))
      setUserA(filled)
    } else {
      setUserA(current.A.map((r) => r.map(() => '')))
    }
  }, [index])

  function handleChange(i, val) {
    const next = inputs.slice(); next[i] = val; setInputs(next); setMessage('')
  }

  function handleAChange(i, j, val) {
    const next = userA.map((r) => r.slice())
    next[i][j] = val
    setUserA(next)
    setMessage('')
  }

  function checkAnswer() {
    // Need exact rational solution based on userA
    // validate A
    const rows = userA.length
    if (rows === 0) { setMessage('Matrix A not defined'); return }
    for (let i = 0; i < rows; i++) for (let j = 0; j < userA[i].length; j++) {
      if (String(userA[i][j]).trim() === '') { setMessage('Please fill all entries of A.'); return }
      if (parseToFraction(String(userA[i][j])) === null) { setMessage('Invalid entry in A: use integers or fractions like 3/2'); return }
    }
    // compute exact solution
    const exact = solveLeastSquaresExact(userA, current.b)
    if (!exact) { setMessage('Could not compute exact solution (singular or invalid A).'); return }
    // parse user answer fractions
    const parsed = inputs.map((s) => parseToFraction(String(s)))
    if (parsed.some((x) => x === null)) { setMessage('Please enter numeric or fractional values for all components.'); return }
    const ok = parsed.every((v, i) => v.equals(exact[i]))
    if (ok) { setMessage('Correct — lock opened!'); return true }
    else { setMessage('Incorrect. Answers must match the exact rational solution.'); return false }
  }

  function onNext() {
    const ok = checkAnswer()
    if (ok) {
      if (index < problems.length - 1) {
        setIndex(index + 1)
      } else {
        setMessage('All puzzles solved — congratulations!')
      }
    }
  }

  function revealSolution() {
    // compute exact solution from user-filled A if possible, otherwise use placeholder A
    const Afor = (userA && userA.length && userA.every((r) => r.every((c) => String(c).trim() !== ''))) ? userA : current.A
    const exact = solveLeastSquaresExact(Afor, current.b)
    if (!exact) { setMessage('Cannot reveal: invalid or singular A.'); return }
    // fill matrix editor with exact/simplified fraction strings
    const filledA = Afor.map((row) => row.map((cell) => {
      if (cell instanceof Fraction) return cell.toString()
      if (typeof cell === 'string') {
        const pf = parseToFraction(cell)
        return pf ? pf.toString() : String(cell)
      }
      if (typeof cell === 'number') {
        const fr = Fraction.fromNumber(cell, 1000)
        return fr ? fr.toString() : String(cell)
      }
      return String(cell)
    }))
    setUserA(filledA)
    setInputs(exact.map((f) => f.toString()))
    setMessage('Solution revealed.')
  }

  return (
    <div id="root">
      <header className="header">
        <h1>Least Squares Escape Room</h1>
        <p className="subtitle">Solve each least squares puzzle to unlock the next room.</p>
      </header>

      <main className="card">
        <div className="progress">Puzzle {index + 1} / {problems.length}</div>
        <h2>{current.title}</h2>
        <p className="desc">{current.desc}</p>

        <div className="matrix">
          {current.editableA === false
            ? <MatrixView matrix={userA && userA.length ? userA : current.A} label="A (fixed)" />
            : <MatrixEditor matrixPlaceholder={current.A} matrixValues={userA} onChange={handleAChange} label="A (editable)" />
          }
          <VectorView vector={current.b} label="b" />
        </div>

        <div className="inputs">
          <label>Enter coefficients (one per box):</label>
          <div className="input-row">
            {inputs.map((v, i) => (
              <input
                key={i}
                value={v}
                onChange={(e) => handleChange(i, e.target.value)}
                placeholder={`x[${i}]`}
              />
            ))}
          </div>
        </div>

        <div className="controls">
          <button className="primary" onClick={onNext}>Next</button>
          <button onClick={() => { setInputs(Array(inputs.length).fill('')); setMessage('') }}>Clear</button>
          <button onClick={revealSolution}>Reveal</button>
        </div>

        <div className="feedback">
          <strong>{message}</strong>
        </div>

        <div className="hint">Tip: the least squares solution solves (A^T A)x = A^T b.</div>
      </main>
    </div>
  )
}

export default App
