import { useEffect, useMemo, useRef, useState } from 'react'

const API = import.meta.env.VITE_API_URL || '/api'

// Party → colour family (muted); matched by substring, case-insensitive.
function partyStyle(party) {
  const p = (party || '').toLowerCase()
  if (/gr[uü]ne|b[uü]ndnis\s*90/.test(p)) return { fg: '#1a6b2e', bg: '#e8f3ea' }
  if (/spd|sozialdemo/.test(p))           return { fg: '#a11221', bg: '#f7e7e9' }
  if (/cdu|csu|union/.test(p))            return { fg: '#1b1b1b', bg: '#ececec' }
  if (/\bfdp\b|freie demokrat/.test(p))   return { fg: '#8a7400', bg: '#fbf6da' }
  if (/linke|\bpds\b/.test(p))            return { fg: '#a1226e', bg: '#f7e6f1' }
  if (/\bafd\b|alternative f/.test(p))    return { fg: '#0a6ea6', bg: '#e3f2fb' }
  return { fg: '#555', bg: '#efefef' }
}

// Highlight query terms in a passage (client-side, escapes HTML).
function highlight(text, query) {
  const terms = (query || '')
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length >= 3)
  if (terms.length === 0) return [text]
  const re = new RegExp('(' + terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi')
  // split() with a capturing group puts the matched terms at odd indices.
  const parts = text.split(re)
  return parts.map((part, i) =>
    i % 2 === 1 ? <mark key={i}>{part}</mark> : <span key={i}>{part}</span>
  )
}

function ResultCard({ row, query }) {
  const [open, setOpen] = useState(false)
  const st = partyStyle(row.party)
  const long = row.text.length > 520
  const shown = open || !long ? row.text : row.text.slice(0, 520).trimEnd() + '…'
  return (
    <article className="card">
      <div className="card-head">
        <span className="speaker">{row.speaker || 'Unbekannt'}</span>
        {row.party && (
          <span className="party" style={{ color: st.fg, background: st.bg }}>{row.party}</span>
        )}
        <span className="spacer" />
        <time className="date">{row.date}</time>
        <span className="score" title="Semantische Ähnlichkeit (Kosinus)">{row.score.toFixed(3)}</span>
      </div>
      <p className="passage">{highlight(shown, query)}</p>
      {long && (
        <button className="more" onClick={() => setOpen((v) => !v)}>
          {open ? 'Weniger anzeigen' : 'Volltext anzeigen'}
        </button>
      )}
    </article>
  )
}

export default function App() {
  const [meta, setMeta] = useState(null)
  const [query, setQuery] = useState('')
  const [yearFrom, setYearFrom] = useState('')
  const [yearTo, setYearTo] = useState('')
  const [topK, setTopK] = useState(10)
  const [results, setResults] = useState(null)
  const [asked, setAsked] = useState('')
  const [took, setTook] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  useEffect(() => {
    fetch(`${API}/meta`)
      .then((r) => r.json())
      .then(setMeta)
      .catch(() => {})
    inputRef.current?.focus()
  }, [])

  const yearHint = meta ? `${meta.min_year}–${meta.max_year}` : ''

  async function runSearch(e, overrideQ) {
    if (e) e.preventDefault()
    const q = (overrideQ != null ? overrideQ : query).trim()
    if (!q || loading) return
    setLoading(true)
    setError(null)
    try {
      const res = await fetch(`${API}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q,
          top_k: Number(topK) || 10,
          year_start: yearFrom ? Number(yearFrom) : null,
          year_end: yearTo ? Number(yearTo) : null,
        }),
      })
      if (!res.ok) throw new Error(`Suche fehlgeschlagen (${res.status})`)
      const data = await res.json()
      setResults(data.results)
      setTook(data.took_ms)
      setAsked(q)
    } catch (err) {
      setError(err.message || 'Unbekannter Fehler')
      setResults(null)
    } finally {
      setLoading(false)
    }
  }

  const examples = useMemo(
    () => ['Deutsche Wiedervereinigung', 'Klimawandel und Energiepolitik', 'Rente mit 63', 'Zuwanderung und Asyl'],
    []
  )

  return (
    <div className="app">
      <header className="masthead">
        <div className="wrap">
          <h1>GermaParl</h1>
          <p className="tagline">Semantische Suche in Reden des Deutschen Bundestags{yearHint && ` · ${yearHint}`}</p>
        </div>
      </header>

      <main className="wrap">
        <form className="search" onSubmit={runSearch}>
          <label htmlFor="q" className="lbl">Wonach suchen Sie?</label>
          <div className="qrow">
            <input
              id="q" ref={inputRef} type="text" value={query} autoComplete="off"
              placeholder="z. B. Deutsche Wiedervereinigung, Klimapolitik, Rente …"
              onChange={(e) => setQuery(e.target.value)}
            />
            <button type="submit" disabled={loading || !query.trim()}>
              {loading ? 'Suche…' : 'Suchen'}
            </button>
          </div>
          <div className="controls">
            <span className="ctl">
              <label>Jahr von</label>
              <input type="number" inputMode="numeric" value={yearFrom} placeholder={meta?.min_year ?? ''}
                     min={meta?.min_year} max={meta?.max_year} onChange={(e) => setYearFrom(e.target.value)} />
            </span>
            <span className="ctl">
              <label>bis</label>
              <input type="number" inputMode="numeric" value={yearTo} placeholder={meta?.max_year ?? ''}
                     min={meta?.min_year} max={meta?.max_year} onChange={(e) => setYearTo(e.target.value)} />
            </span>
            <span className="ctl">
              <label>Treffer</label>
              <select value={topK} onChange={(e) => setTopK(e.target.value)}>
                {[5, 10, 20, 30, 50].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </span>
          </div>
          {results === null && !loading && (
            <p className="hint">
              Beispiele:{' '}
              {examples.map((ex, i) => (
                <button type="button" key={ex} className="chip"
                        onClick={() => { setQuery(ex); runSearch(null, ex) }}>
                  {ex}
                </button>
              ))}
            </p>
          )}
        </form>

        {error && <div className="notice error">{error}</div>}

        {results !== null && (
          <section className="results">
            <p className="rmeta">
              {results.length > 0
                ? <>{results.length} Reden zu „{asked}“, nach semantischer Ähnlichkeit sortiert · {took} ms</>
                : <>Keine Treffer für „{asked}“{(yearFrom || yearTo) ? ' im gewählten Zeitraum' : ''}.</>}
            </p>
            {results.map((row, i) => (
              <ResultCard key={`${row.date}-${i}`} row={row} query={asked} />
            ))}
          </section>
        )}
      </main>

      <footer className="foot wrap">
        <p>
          Volltextnahe semantische Suche über {meta ? meta.n_speeches.toLocaleString('de-DE') : '≈1 Mio.'} Redebeiträge.
          Nur Retrieval — es werden ähnlichkeitssortierte Originalreden angezeigt, keine generierte Zusammenfassung.
        </p>
        <p className="src">
          Datenbasis: GermaParl (PolMine, Blätte et&nbsp;al.). Embeddings: jina-embeddings-v4. Prüfen Sie Treffer stets am Originalprotokoll.
        </p>
      </footer>
    </div>
  )
}
