import { useEffect, useRef, useState } from 'react'

const API = import.meta.env.VITE_API_URL || '/api'
const CODE_URL = 'https://github.com/KonstantinWandel/parliamentary-rag'
// Concept DOI: it resolves to whatever the newest archived release is, so the footer stays
// correct across releases. The DOI of a single release is in CITATION.cff.
const DOI = import.meta.env.VITE_DOI || '10.5281/zenodo.21194707'

const UI = {
  h1: 'Plenarsuche',
  title: 'Plenarsuche: Semantische Suche in Bundestagsreden',
  tagline: 'Semantische Suche in Reden des Deutschen Bundestags',
  placeholder: 'z. B. Deutsche Wiedervereinigung, Klimapolitik, Rente …',
  examples: ['Deutsche Wiedervereinigung', 'Klimawandel und Energiepolitik', 'Rente mit 63', 'Zuwanderung und Asyl'],
  model: 'jina-embeddings-v4',
  // Credit and licence of the corpus, shown in the footer and on every exported PDF. The prepared
  // GermaParl corpus is published under CLARIN PUB+BY+NC+SA (github.com/PolMine/GermaParlTEI).
  source: 'GermaParl (PolMine, Blätte et al.), Lizenz CLARIN PUB+BY+NC+SA.',
}

// An empty year field shows its limit as a placeholder (1949, 2025), but the browser treats it as
// having no value and sent the first arrow press somewhere else: from the shown 2025, "bis" jumped
// to 1949. The shown limit is used as the starting value instead, so one press moves from it by
// one year, and the result stays inside the corpus years.
function stepFromPlaceholder(e, value, setValue, start, lo, hi) {
  if (value !== '' || start == null || lo == null || hi == null) return
  if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return
  e.preventDefault()
  const next = start + (e.key === 'ArrowUp' ? 1 : -1)
  setValue(String(Math.min(hi, Math.max(lo, next))))
}

// One speech, one PDF. jsPDF's built-in Helvetica only covers Latin-1 and silently drops other
// characters ("Dağdelen" came out as "Dadelen"), so a subset of DejaVu Sans is embedded; the
// two files are fetched on the first export. If they cannot be fetched, Helvetica is used.
const PAGE = { width: 595.28, height: 841.89, margin: 56 }
const FONT_FILES = { normal: '/fonts/DejaVuSans-subset.ttf', bold: '/fonts/DejaVuSans-Bold-subset.ttf' }
let fontCache = null

async function loadFonts() {
  if (fontCache) return fontCache
  try {
    const entries = await Promise.all(Object.entries(FONT_FILES).map(async ([style, url]) => {
      const res = await fetch(url)
      if (!res.ok) throw new Error(url)
      const bytes = new Uint8Array(await res.arrayBuffer())
      let bin = ''
      for (let i = 0; i < bytes.length; i += 0x8000) bin += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
      return [style, btoa(bin)]
    }))
    fontCache = Object.fromEntries(entries)
  } catch {
    fontCache = {}
  }
  return fontCache
}

function fileSafe(value) {
  return (value || '')
    .replace(/[ä]/gi, 'ae').replace(/[ö]/gi, 'oe').replace(/[ü]/gi, 'ue').replace(/ß/g, 'ss')
    .normalize('NFD').replace(/[̀-ͯ]/g, '')
    .replace(/[^\w.-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^_|_$/g, '')
    .slice(0, 60)
}

// Plain-text and CSV export of the selected speeches, one file each. Both carry the source and the
// licence, because the corpus licence asks for attribution wherever the texts travel.
function download(content, filename, mime) {
  const url = URL.createObjectURL(new Blob([content], { type: mime }))
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  setTimeout(() => URL.revokeObjectURL(url), 2000)
}

function exportName(asked, ext) {
  const day = new Date().toISOString().slice(0, 10)
  return `plenarsuche_${fileSafe(asked) || 'suche'}_${day}.${ext}`
}

function buildTxt(rows, asked) {
  const stamp = new Date().toLocaleDateString('de-DE')
  const head = [
    'Plenarsuche, Export vom ' + stamp,
    asked ? `Suchanfrage: „${asked}“` : null,
    `Quelle: ${UI.source}`,
    `Anzahl Reden: ${rows.length}`,
  ].filter(Boolean).join('\n')
  const rule = '='.repeat(72)
  const body = rows.map((row, i) => {
    const who = [row.speaker || 'Unbekannt', row.party ? `(${row.party})` : ''].filter(Boolean).join(' ')
    return `${rule}\n${i + 1}. ${who}, ${row.date}, Ähnlichkeit ${row.score.toFixed(3)}\n${'-'.repeat(72)}\n${String(row.text || '').trim()}\n`
  }).join('\n')
  return `${head}\n\n${body}`
}

// A cell that starts with = + - @ would be read as a formula by a spreadsheet, so it gets a leading apostrophe.
function csvCell(value) {
  let s = String(value ?? '')
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`
  return `"${s.replace(/"/g, '""')}"`
}

function buildCsv(rows, asked) {
  const cols = ['rank', 'score', 'date', 'year', 'speaker', 'party', 'text', 'query', 'source']
  const lines = [cols.join(',')]
  rows.forEach((row, i) => {
    lines.push([i + 1, row.score, row.date, row.year, row.speaker, row.party, row.text, asked, UI.source].map(csvCell).join(','))
  })
  return '\ufeff' + lines.join('\r\n') + '\r\n'   // byte order mark so Excel reads the file as UTF-8
}

function speechPdf(jsPDF, fonts, row, asked) {
  const doc = new jsPDF({ unit: 'pt', format: 'a4' })
  let face = 'helvetica'
  if (fonts.normal && fonts.bold) {
    doc.addFileToVFS('DejaVuSans.ttf', fonts.normal)
    doc.addFont('DejaVuSans.ttf', 'DejaVu', 'normal')
    doc.addFileToVFS('DejaVuSans-Bold.ttf', fonts.bold)
    doc.addFont('DejaVuSans-Bold.ttf', 'DejaVu', 'bold')
    face = 'DejaVu'
  }
  const inner = PAGE.width - 2 * PAGE.margin
  // The title is what the reader needs to identify the speech: who said it, for which party, and when.
  const who = row.speaker || 'Unbekannt'
  const affiliation = row.party || ''
  const subtitle = [affiliation, row.date].filter(Boolean).join(' · ')

  let y = PAGE.margin
  doc.setFont(face, 'bold').setFontSize(16)
  for (const line of doc.splitTextToSize(who, inner)) {
    doc.text(line, PAGE.margin, y)
    y += 21
  }
  if (subtitle) {
    doc.setFont(face, 'normal').setFontSize(10.5).setTextColor(90)
    doc.text(subtitle, PAGE.margin, y)
    y += 16
  }
  y += 6
  doc.setDrawColor(180).line(PAGE.margin, y, PAGE.width - PAGE.margin, y)
  y += 22

  doc.setFont(face, 'normal').setFontSize(10.5).setTextColor(20)
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
  const note = `${UI.source} Export ${stamp}${asked ? ` · Suchanfrage: „${asked}“` : ''}`
  const pages = doc.getNumberOfPages()
  for (let page = 1; page <= pages; page += 1) {
    doc.setPage(page)
    doc.setFont(face, 'normal').setFontSize(7.5).setTextColor(110)
    doc.text(doc.splitTextToSize(note, inner - 40), PAGE.margin, PAGE.height - 34)
    doc.text(`${page}/${pages}`, PAGE.width - PAGE.margin, PAGE.height - 34, { align: 'right' })
  }

  // Several speeches of one speaker on one day are common, so the rank keeps the file names apart.
  const name = [row.date, fileSafe(who), fileSafe(affiliation), row.rank ? `Treffer${row.rank}` : ''].filter(Boolean).join('_')
  return { doc, filename: `${name || 'rede'}.pdf` }
}

function partyStyle(party) {
  const p = (party || '').toLowerCase()
  if (/gr[uü]ne|b[uü]ndnis\s*90/.test(p)) return { fg: '#1a6b2e', bg: '#e8f3ea' }
  if (/spd|sozialdemo/.test(p))           return { fg: '#a11221', bg: '#f7e7e9' }
  if (/cdu|csu|union/.test(p))            return { fg: '#1b1b1b', bg: '#ececec' }
  if (/\bfdp\b|freie demokrat/.test(p))   return { fg: '#756200', bg: '#fbf6da' }
  if (/linke|\bpds\b/.test(p))            return { fg: '#a1226e', bg: '#f7e6f1' }
  if (/\bafd\b|alternative f/.test(p))    return { fg: '#0a6ea6', bg: '#e3f2fb' }
  return { fg: '#555', bg: '#efefef' }
}

// Function words are not marked, and a term matches from the start of a word ("Rente" marks
// "Renteneintrittsalter", but "mit" does not mark the inside of "Mitglied" or "damit").
const STOPWORDS = new Set(['und', 'oder', 'der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine', 'einen', 'einem', 'einer',
  'mit', 'von', 'vom', 'zum', 'zur', 'für', 'auf', 'aus', 'bei', 'nach', 'über', 'unter', 'als', 'auch', 'nicht', 'sich',
  'ist', 'sind', 'war', 'wird', 'werden', 'wir', 'ich', 'sie', 'wie', 'was', 'wer', 'im', 'in', 'an', 'zu', 'um', 'es'])

function highlight(text, query) {
  const terms = (query || '').toLowerCase().split(/[^\p{L}\p{N}]+/u).filter((t) => t.length >= 3 && !STOPWORDS.has(t))
  if (terms.length === 0) return [text]
  // Split into words and non-word runs; a word that starts with a term gets that start marked.
  return text.split(/([\p{L}\p{N}]+)/u).map((part, i) => {
    if (i % 2 === 0) return <span key={i}>{part}</span>
    const low = part.toLowerCase()
    const hit = terms.find((t) => low.startsWith(t))
    return hit
      ? <span key={i}><mark>{part.slice(0, hit.length)}</mark>{part.slice(hit.length)}</span>
      : <span key={i}>{part}</span>
  })
}

function ResultCard({ row, query, picked, onPick }) {
  const [open, setOpen] = useState(false)
  const long = row.text.length > 520
  const shown = open || !long ? row.text : row.text.slice(0, 520).trimEnd() + '…'
  const st = row.party ? partyStyle(row.party) : null
  return (
    <article className={picked ? 'card is-picked' : 'card'}>
      <div className="card-head">
        <input type="checkbox" className="pick" checked={picked} onChange={onPick}
               aria-label={`Rede von ${row.speaker || 'unbekannt'} vom ${row.date} auswählen`} />
        <span className="speaker">{row.speaker || 'Unbekannt'}</span>
        {st && <span className="party" style={{ color: st.fg, background: st.bg }}>{row.party}</span>}
        <span className="spacer" />
        <time className="date">{row.date}</time>
        <span className="score" title="Semantische Ähnlichkeit (Kosinus)">{row.score.toFixed(3)}</span>
      </div>
      <p className="passage">{highlight(shown, query)}</p>
      {long && (
        <button className="more" aria-expanded={open} onClick={() => setOpen((v) => !v)}>
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
  const [picked, setPicked] = useState({})
  const [exporting, setExporting] = useState(false)
  const inputRef = useRef(null)

  useEffect(() => {
    document.title = UI.title
    inputRef.current?.focus()
    fetch(`${API}/meta`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then(setMeta).catch(() => {})
  }, [])

  const yearHint = meta ? `${meta.min_year}–${meta.max_year}` : ''

  async function runSearch(e, overrideQ) {
    if (e) e.preventDefault()
    const q = (overrideQ != null ? overrideQ : query).trim()
    if (!q || loading) return
    if (yearFrom && yearTo && Number(yearFrom) > Number(yearTo)) {
      setError('Das Startjahr liegt nach dem Endjahr. Bitte den Zeitraum anpassen.'); return
    }
    setLoading(true); setError(null)
    try {
      const res = await fetch(`${API}/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q, top_k: Number(topK) || 10,
          year_start: yearFrom ? Number(yearFrom) : null,
          year_end: yearTo ? Number(yearTo) : null,
        }),
      })
      if (!res.ok) {
        let msg = `Suche fehlgeschlagen (${res.status})`
        try { const j = await res.json(); if (j && typeof j.detail === 'string') msg = j.detail } catch { /* keep default */ }
        throw new Error(msg)
      }
      const data = await res.json()
      setResults(data.results); setTook(data.took_ms); setAsked(q); setPicked({})
    } catch (err) {
      // fetch() rejects with a bare TypeError ("Failed to fetch") when the server cannot be reached.
      setError(err instanceof TypeError ? 'Der Server ist momentan nicht erreichbar. Bitte gleich erneut versuchen.' : (err.message || 'Unbekannter Fehler'))
      setResults(null)
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
      const [{ jsPDF }, fonts] = await Promise.all([import('jspdf'), loadFonts()])   // loaded on first use
      for (const row of pickedRows) {
        const { doc, filename } = speechPdf(jsPDF, fonts, row, asked)
        doc.save(filename)
        await new Promise((resolve) => setTimeout(resolve, 350))
      }
    } finally {
      setExporting(false)
    }
  }

  function downloadTxt() {
    if (pickedRows.length) download(buildTxt(pickedRows, asked), exportName(asked, 'txt'), 'text/plain;charset=utf-8')
  }

  function downloadCsv() {
    if (pickedRows.length) download(buildCsv(pickedRows, asked), exportName(asked, 'csv'), 'text/csv;charset=utf-8')
  }

  return (
    <div className="app">
      <header className="masthead">
        <div className="wrap masthead-row">
          <div className="brand">
            <h1>{UI.h1}</h1>
            <p className="tagline">{UI.tagline}{yearHint && ` · ${yearHint}`}</p>
          </div>
          <div className="masthead-actions">
            <a className="header-feedback" href="https://geolab.soz.uni-bielefeld.de/feedback/?app=germaparl" referrerPolicy="no-referrer">Feedback geben</a>
          </div>
        </div>
      </header>

      <main className="wrap">
        <form className="search" onSubmit={runSearch}>
          <label htmlFor="q" className="lbl">Wonach suchen Sie?</label>
          <div className="qrow">
            <input
              id="q" ref={inputRef} type="text" value={query} autoComplete="off" maxLength={500}
              placeholder={UI.placeholder} onChange={(e) => setQuery(e.target.value)}
            />
            <button type="submit" disabled={loading || !query.trim()}>{loading ? 'Suche…' : 'Suchen'}</button>
          </div>
          <div className="controls">
            <span className="ctl">
              <label htmlFor="year-from">Jahr von</label>
              <input id="year-from" type="number" inputMode="numeric" value={yearFrom} placeholder={meta?.min_year ?? ''}
                     min={meta?.min_year} max={meta?.max_year} onChange={(e) => setYearFrom(e.target.value)}
                     onKeyDown={(e) => stepFromPlaceholder(e, yearFrom, setYearFrom, meta?.min_year, meta?.min_year, meta?.max_year)} />
            </span>
            <span className="ctl">
              <label htmlFor="year-to">bis</label>
              <input id="year-to" type="number" inputMode="numeric" value={yearTo} placeholder={meta?.max_year ?? ''}
                     min={meta?.min_year} max={meta?.max_year} onChange={(e) => setYearTo(e.target.value)}
                     onKeyDown={(e) => stepFromPlaceholder(e, yearTo, setYearTo, meta?.max_year, meta?.min_year, meta?.max_year)} />
            </span>
            <span className="ctl">
              <label htmlFor="top-k">Treffer</label>
              <select id="top-k" value={topK} onChange={(e) => setTopK(e.target.value)}>
                {[5, 10, 20, 30, 50].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </span>
          </div>
          {results === null && !loading && (
            <p className="hint">
              Beispiele:{' '}
              {UI.examples.map((ex) => (
                <button type="button" key={ex} className="chip"
                        onClick={() => { setQuery(ex); runSearch(null, ex) }}>{ex}</button>
              ))}
            </p>
          )}
        </form>

        {error && <div className="notice error" role="alert">{error}</div>}

        {results !== null && (
          <section className="results">
            <p className="rmeta" aria-live="polite">
              {results.length > 0
                ? <>{results.length} Reden zu „{asked}“, nach semantischer Ähnlichkeit sortiert · {took} ms</>
                : <>Keine Treffer für „{asked}“{(yearFrom || yearTo) ? ' im gewählten Zeitraum' : ''}.</>}
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
                <span className="exports" role="group" aria-label="Ausgewählte Reden herunterladen">
                  <button type="button" className="pdfbtn" disabled={!pickedRows.length || exporting}
                          onClick={downloadPdfs} title="Eine PDF-Datei pro Rede">
                    {exporting ? 'PDFs werden erzeugt…' : 'PDF (je Rede)'}
                  </button>
                  <button type="button" className="pdfbtn" disabled={!pickedRows.length}
                          onClick={downloadTxt} title="Alle ausgewählten Reden in einer Textdatei">
                    TXT
                  </button>
                  <button type="button" className="pdfbtn" disabled={!pickedRows.length}
                          onClick={downloadCsv} title="Alle ausgewählten Reden in einer CSV-Tabelle (UTF-8)">
                    CSV
                  </button>
                </span>
              </div>
            )}
            {results.map((row, i) => (
              <ResultCard key={`${asked}-${row.date}-${i}`} row={row} query={asked}
                          picked={Boolean(picked[i])}
                          onPick={() => setPicked((c) => ({ ...c, [i]: !c[i] }))} />
            ))}
          </section>
        )}
      </main>

      <footer className="foot wrap">
        <p>
          Semantische Suche über rund {meta ? (Number(meta.count) / 1e6).toLocaleString('de-DE', { maximumFractionDigits: 2 }) : '…'} Millionen Reden.
          Angezeigt werden die Originaltreffer, nach Ähnlichkeit sortiert; eine generierte Zusammenfassung gibt es nicht.
        </p>
        <p className="src">Datenbasis: {UI.source} Embeddings: {UI.model}. Treffer stets an der Quelle prüfen.</p>
        <p className="cite">
          Zitiervorschlag: Wandel, K. (2026). <em>Plenarsuche</em>. Universität Bielefeld.{' '}
          <a href={CODE_URL} target="_blank" rel="noopener noreferrer">Quellcode auf GitHub</a>
          {DOI && <> · <a href={`https://doi.org/${DOI}`} target="_blank" rel="noopener noreferrer">doi.org/{DOI}</a></>}
        </p>
      </footer>
    </div>
  )
}
