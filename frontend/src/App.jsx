import { useEffect, useRef, useState } from 'react'
import { jsPDF } from 'jspdf'

const API = import.meta.env.VITE_API_URL || '/api'
const CODE_URL = 'https://github.com/KonstantinWandel/parliamentary-rag'
// Concept DOI: it resolves to whatever the newest archived release is, so the footer stays
// correct across releases. The DOI of a single release is in CITATION.cff.
const DOI = import.meta.env.VITE_DOI || '10.5281/zenodo.21194707'

// The two universities behind the tool and the corpus, the funder, and the project that
// publishes GermaParl. Monochrome SVGs are drawn as masks so they follow the text colour;
// the two colour logos are served as files from this site, never hotlinked.
const BRANDS = [
  { file: 'uni-bielefeld.svg', alt: 'Universität Bielefeld', mask: true, shape: 'brand-uni',
    url: 'https://www.uni-bielefeld.de/' },
  { file: 'leibniz.svg', alt: 'Leibniz-Gemeinschaft', mask: true, shape: 'brand-leibniz',
    url: 'https://www.leibniz-gemeinschaft.de/' },
  { file: 'polmine.png', alt: 'PolMine (GermaParl)', mask: false, shape: 'brand-polmine',
    url: 'https://polmine.github.io/' },
  { file: 'uni-duisburg-essen.png', alt: 'Universität Duisburg-Essen', mask: false,
    shape: 'brand-ude', url: 'https://www.uni-due.de/politik/blaette.php' },
]

// Per-corpus UI copy, keyed by the backend `kind`.
const UI = {
  parliament: {
    h1: 'GermaParl',
    tagline: 'Semantische Suche in Reden des Deutschen Bundestags',
    placeholder: 'z. B. Deutsche Wiedervereinigung, Klimapolitik, Rente …',
    examples: ['Deutsche Wiedervereinigung', 'Klimawandel und Energiepolitik', 'Rente mit 63', 'Zuwanderung und Asyl'],
    noun: 'Reden',
    model: 'jina-embeddings-v4',
    source: 'GermaParl (PolMine, Blätte et al.).',
  },
  press: {
    h1: 'RegioPress',
    tagline: 'Semantische Suche in regionalen Zeitungsartikeln',
    placeholder: 'z. B. Hochwasserschutz, Innenstadtentwicklung, Windkraft …',
    examples: ['Hochwasserschutz', 'Fachkräftemangel', 'Windkraft vor Ort', 'Innenstadt und Leerstand'],
    noun: 'Artikel',
    model: 'multilingual-e5-large-instruct',
    source: 'RegioPress (Genios). Lizenzpflichtig — nur für berechtigte Nutzer.',
  },
}

// One speech, one PDF. jsPDF's built-in Helvetica is WinAnsi-encoded, which covers German
// umlauts and the typographic quotes the corpus uses, so no font has to be embedded.
const PAGE = { width: 595.28, height: 841.89, margin: 56 }

function fileSafe(value) {
  return (value || '')
    .replace(/[ä]/gi, 'ae').replace(/[ö]/gi, 'oe').replace(/[ü]/gi, 'ue').replace(/ß/g, 'ss')
    .replace(/[^\w.-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '')
    .slice(0, 60)
}

function speechPdf(row, kind, ui, asked) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' })
  const inner = PAGE.width - 2 * PAGE.margin
  // The title is what the reader needs to identify the speech: who said it, for which party,
  // and when. For the press corpus the same three slots hold paper, section and date.
  const who = (kind === 'press' ? row.source : row.speaker) || 'Unbekannt'
  const affiliation = (kind === 'press' ? row.ressort : row.party) || ''
  const subtitle = [affiliation, row.date].filter(Boolean).join(' · ')

  let y = PAGE.margin
  doc.setFont('helvetica', 'bold').setFontSize(17)
  for (const line of doc.splitTextToSize(who, inner)) {
    doc.text(line, PAGE.margin, y)
    y += 21
  }
  if (subtitle) {
    doc.setFont('helvetica', 'normal').setFontSize(11).setTextColor(90)
    doc.text(subtitle, PAGE.margin, y)
    y += 16
  }
  if (kind === 'press' && row.title) {
    doc.setFont('helvetica', 'italic').setFontSize(12).setTextColor(40)
    for (const line of doc.splitTextToSize(row.title, inner)) {
      doc.text(line, PAGE.margin, y)
      y += 16
    }
  }
  y += 6
  doc.setDrawColor(180).line(PAGE.margin, y, PAGE.width - PAGE.margin, y)
  y += 22

  doc.setFont('helvetica', 'normal').setFontSize(11).setTextColor(20)
  const body = doc.splitTextToSize(String(row.text || '').replace(/\s+\n/g, '\n'), inner)
  for (const line of body) {
    if (y > PAGE.height - PAGE.margin - 30) {
      doc.addPage()
      y = PAGE.margin
    }
    doc.text(line, PAGE.margin, y)
    y += 15.5
  }

  const stamp = new Date().toLocaleDateString('de-DE')
  const note = `${ui.source} Export ${stamp}${asked ? ` · Suchanfrage: "${asked}"` : ''}`
  const pages = doc.getNumberOfPages()
  for (let page = 1; page <= pages; page += 1) {
    doc.setPage(page)
    doc.setFont('helvetica', 'normal').setFontSize(8).setTextColor(130)
    doc.text(doc.splitTextToSize(note, inner - 40)[0], PAGE.margin, PAGE.height - 28)
    doc.text(`${page}/${pages}`, PAGE.width - PAGE.margin, PAGE.height - 28, { align: 'right' })
  }

  const name = [row.date, fileSafe(who), fileSafe(affiliation)].filter(Boolean).join('_')
  return { doc, filename: `${name || 'rede'}.pdf` }
}

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

function highlight(text, query) {
  const terms = (query || '').toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((t) => t.length >= 3)
  if (terms.length === 0) return [text]
  const re = new RegExp('(' + terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|') + ')', 'gi')
  return text.split(re).map((part, i) =>
    i % 2 === 1 ? <mark key={i}>{part}</mark> : <span key={i}>{part}</span>
  )
}

function ResultCard({ row, query, kind, picked, onPick }) {
  const [open, setOpen] = useState(false)
  const long = row.text.length > 520
  const shown = open || !long ? row.text : row.text.slice(0, 520).trimEnd() + '…'
  const head = kind === 'press' ? (
    <div className="card-head">
      <input type="checkbox" className="pick" checked={picked} onChange={onPick}
             aria-label={`${row.source || 'Treffer'} vom ${row.date} auswählen`} />
      <span className="speaker">{row.source || 'Unbekannt'}</span>
      {row.ressort && <span className="tag">{row.ressort}</span>}
      <span className="spacer" />
      <time className="date">{row.date}</time>
      <span className="score" title="Semantische Ähnlichkeit (Kosinus)">{row.score.toFixed(3)}</span>
    </div>
  ) : (
    <div className="card-head">
      <input type="checkbox" className="pick" checked={picked} onChange={onPick}
             aria-label={`Rede von ${row.speaker || 'unbekannt'} vom ${row.date} auswählen`} />
      <span className="speaker">{row.speaker || 'Unbekannt'}</span>
      {row.party && (() => { const st = partyStyle(row.party); return (
        <span className="party" style={{ color: st.fg, background: st.bg }}>{row.party}</span>
      ) })()}
      <span className="spacer" />
      <time className="date">{row.date}</time>
      <span className="score" title="Semantische Ähnlichkeit (Kosinus)">{row.score.toFixed(3)}</span>
    </div>
  )
  return (
    <article className={picked ? 'card is-picked' : 'card'}>
      {head}
      {kind === 'press' && row.title && <p className="headline">{highlight(row.title, query)}</p>}
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
  const [corpora, setCorpora] = useState([])
  const [corpus, setCorpus] = useState('')
  const [meta, setMeta] = useState(null)
  const [query, setQuery] = useState('')
  const [yearFrom, setYearFrom] = useState('')
  const [yearTo, setYearTo] = useState('')
  const [sourceId, setSourceId] = useState('')
  const [ressort, setRessort] = useState('')
  const [topK, setTopK] = useState(10)
  const [results, setResults] = useState(null)
  const [asked, setAsked] = useState('')
  const [took, setTook] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [picked, setPicked] = useState({})
  const [exporting, setExporting] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    fetch(`${API}/corpora`)
      .then((r) => r.json())
      .then((d) => { setCorpora(d.corpora || []); setCorpus(d.default || d.corpora?.[0]?.id || '') })
      .catch(() => {})
    inputRef.current?.focus()
  }, [])

  useEffect(() => {
    if (!corpus) return
    setResults(null); setError(null); setSourceId(''); setRessort(''); setYearFrom(''); setYearTo('')
    fetch(`${API}/meta?corpus=${encodeURIComponent(corpus)}`)
      .then((r) => r.json()).then(setMeta).catch(() => {})
  }, [corpus])

  const kind = meta?.kind || corpora.find((c) => c.id === corpus)?.kind || 'parliament'
  const ui = UI[kind] || UI.parliament

  // index.html carries one static title for both deployments, so the GermaParl host showed
  // "RegioPress" in the browser tab. The corpus decides it.
  useEffect(() => {
    document.title = `${ui.h1} — ${ui.tagline}`
  }, [ui.h1, ui.tagline])
  const yearHint = meta ? `${meta.min_year}–${meta.max_year}` : ''

  async function runSearch(e, overrideQ) {
    if (e) e.preventDefault()
    const q = (overrideQ != null ? overrideQ : query).trim()
    if (!q || loading) return
    setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q, corpus, top_k: Number(topK) || 10,
          year_start: yearFrom ? Number(yearFrom) : null,
          year_end: yearTo ? Number(yearTo) : null,
          source_id: kind === 'press' && sourceId ? sourceId : null,
          ressort: kind === 'press' && ressort ? ressort : null,
        }),
      })
      if (!res.ok) {
        let msg = `Suche fehlgeschlagen (${res.status})`
        try { const j = await res.json(); if (j && j.detail) msg = j.detail } catch { /* keep default */ }
        throw new Error(msg)
      }
      const data = await res.json()
      setResults(data.results); setTook(data.took_ms); setAsked(q); setPicked({})
    } catch (err) {
      setError(err.message || 'Unbekannter Fehler'); setResults(null)
    } finally {
      setLoading(false)
    }
  }

  const pickedRows = (results || []).filter((_, i) => picked[i])

  // One PDF per selected speech. Browsers rate-limit a burst of downloads, so they are spaced
  // out; Chrome asks once whether the site may download several files.
  async function downloadPdfs() {
    if (!pickedRows.length || exporting) return
    setExporting(true)
    try {
      for (const row of pickedRows) {
        const { doc, filename } = speechPdf(row, kind, ui, asked)
        doc.save(filename)
        await new Promise((resolve) => setTimeout(resolve, 350))
      }
    } finally {
      setExporting(false)
    }
  }

  return (
    <div className="app">
      <header className="masthead">
        <div className="wrap masthead-row">
          <div className="brand">
            <h1>{ui.h1}</h1>
            <p className="tagline">{ui.tagline}{yearHint && ` · ${yearHint}`}</p>
          </div>
          {corpora.length > 0 && (
            <label className="dataset">
              <span>Datensatz</span>
              <select value={corpus} onChange={(e) => setCorpus(e.target.value)}>
                {corpora.map((c) => <option key={c.id} value={c.id}>{c.label}</option>)}
              </select>
            </label>
          )}
        </div>
      </header>

      <main className="wrap">
        <form className="search" onSubmit={runSearch}>
          <label htmlFor="q" className="lbl">Wonach suchen Sie?</label>
          <div className="qrow">
            <input
              id="q" ref={inputRef} type="text" value={query} autoComplete="off"
              placeholder={ui.placeholder} onChange={(e) => setQuery(e.target.value)}
            />
            <button type="submit" disabled={loading || !query.trim()}>{loading ? 'Suche…' : 'Suchen'}</button>
          </div>
          <div className="controls">
            {kind === 'press' && meta?.sources?.length > 0 && (
              <span className="ctl">
                <label>Zeitung</label>
                <select value={sourceId} onChange={(e) => setSourceId(e.target.value)}>
                  <option value="">alle</option>
                  {meta.sources.map((s) => <option key={s.id} value={s.id}>{s.name}</option>)}
                </select>
              </span>
            )}
            {kind === 'press' && meta?.ressorts?.length > 0 && (
              <span className="ctl">
                <label>Ressort</label>
                <select value={ressort} onChange={(e) => setRessort(e.target.value)}>
                  <option value="">alle</option>
                  {meta.ressorts.map((r) => <option key={r} value={r}>{r}</option>)}
                </select>
              </span>
            )}
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
              {ui.examples.map((ex) => (
                <button type="button" key={ex} className="chip"
                        onClick={() => { setQuery(ex); runSearch(null, ex) }}>{ex}</button>
              ))}
            </p>
          )}
        </form>

        {error && <div className="notice error">{error}</div>}

        {results !== null && (
          <section className="results">
            <p className="rmeta">
              {results.length > 0
                ? <>{results.length} {ui.noun} zu „{asked}“, nach semantischer Ähnlichkeit sortiert · {took} ms</>
                : <>Keine Treffer für „{asked}“{(yearFrom || yearTo || sourceId || ressort) ? ' mit den gewählten Filtern' : ''}.</>}
            </p>
            {results.length > 0 && (
              <div className="picktools">
                <label className="pickall">
                  <input
                    type="checkbox"
                    checked={pickedRows.length === results.length && results.length > 0}
                    onChange={(e) => setPicked(e.target.checked
                      ? Object.fromEntries(results.map((_, i) => [i, true]))
                      : {})}
                  />
                  Alle auswählen
                </label>
                <span className="pickcount">
                  {pickedRows.length > 0 ? `${pickedRows.length} ausgewählt` : 'nichts ausgewählt'}
                </span>
                <button type="button" className="pdfbtn" disabled={!pickedRows.length || exporting}
                        onClick={downloadPdfs}>
                  {exporting ? 'PDFs werden erzeugt…' : 'Als PDF herunterladen'}
                </button>
              </div>
            )}
            {results.map((row, i) => (
              <ResultCard key={`${row.date}-${i}`} row={row} query={asked} kind={kind}
                          picked={Boolean(picked[i])}
                          onPick={() => setPicked((c) => ({ ...c, [i]: !c[i] }))} />
            ))}
          </section>
        )}
      </main>

      <footer className="foot wrap">
        <p>
          Volltextnahe semantische Suche über {meta ? Number(meta.count).toLocaleString('de-DE') : '…'} {ui.noun}.
          Nur Retrieval — ähnlichkeitssortierte Originaltreffer, keine generierte Zusammenfassung.
        </p>
        <p className="src">Datenbasis: {ui.source} Embeddings: {ui.model}. Treffer stets an der Quelle prüfen.</p>
        <p className="cite">
          Zitiervorschlag: Wandel, K. (2026). <em>Parliamentary Speech Finder</em>. Universität Bielefeld.{' '}
          <a href={CODE_URL} target="_blank" rel="noopener noreferrer">Quellcode auf GitHub</a>
          {DOI && <> · <a href={`https://doi.org/${DOI}`} target="_blank" rel="noopener noreferrer">doi.org/{DOI}</a></>}
        </p>
      </footer>

      <div className="brand-strip">
        {BRANDS.map((brand) => (
          <a key={brand.file} href={brand.url} target="_blank" rel="noopener noreferrer"
             title={brand.alt} aria-label={brand.alt}>
            {brand.mask ? (
              <span role="img" aria-label={brand.alt} className={`brand-logo brand-mark ${brand.shape}`} />
            ) : (
              <img src={`/brand/${brand.file}`} alt={brand.alt} className={`brand-logo ${brand.shape}`}
                   onError={(e) => { e.currentTarget.parentElement.style.display = 'none' }} />
            )}
          </a>
        ))}
      </div>
    </div>
  )
}
