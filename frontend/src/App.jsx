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
  // Shown only while the server can generate answers; a click runs the question with the answer switched on.
  questions: [
    'Wie wurde im Bundestag über den Mindestlohn argumentiert?',
    'Welche Argumente gab es für und gegen den Atomausstieg?',
    'Wie haben Rednerinnen und Redner die Wiedervereinigung bewertet?',
    'Wie wurde die Einführung des Euro begründet?',
  ],
  model: 'jina-embeddings-v4',
  // Credit and licence of the corpus, shown in the footer and on every exported PDF. The prepared
  // GermaParl corpus is published under CLARIN PUB+BY+NC+SA (github.com/PolMine/GermaParlTEI).
  source: 'GermaParl (PolMine, Blätte et al.), Lizenz CLARIN PUB+BY+NC+SA.',
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

function ResultCard({ row, query, picked, onPick, id, numbered }) {
  const [open, setOpen] = useState(false)
  const long = row.text.length > 520
  const shown = open || !long ? row.text : row.text.slice(0, 520).trimEnd() + '…'
  const st = row.party ? partyStyle(row.party) : null
  return (
    <article id={id} className={picked ? 'card is-picked' : 'card'}>
      <div className="card-head">
        <input type="checkbox" className="pick" checked={picked} onChange={onPick}
               aria-label={`Rede von ${row.speaker || 'unbekannt'} vom ${row.date} auswählen`} />
        {numbered && <span className="rank" title="Nummer, mit der die Antwort diese Rede zitiert">[{row.rank}]</span>}
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

// The model cites sources as [n]. Each reference becomes a button that scrolls to that result card.
// The text is rendered as plain text nodes, never as HTML.
function AnswerText({ text }) {
  const jump = (n) => {
    const el = document.getElementById(`src-${n}`)
    if (!el) return
    el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    el.classList.add('flash')
    setTimeout(() => el.classList.remove('flash'), 1600)
  }
  return text.split(/\n{2,}/).map((para, pi) => (
    <p key={pi}>
      {para.split(/(\[\d{1,2}\])/).map((part, i) => {
        const m = /^\[(\d{1,2})\]$/.exec(part)
        return m
          ? <button key={i} type="button" className="cite-ref" onClick={() => jump(m[1])}
                    aria-label={`Zu Quelle ${m[1]}`}>{m[1]}</button>
          : <span key={i}>{part}</span>
      })}
    </p>
  ))
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
  const [answerOK, setAnswerOK] = useState(false)      // the server can generate answers right now
  const [withAnswer, setWithAnswer] = useState(false)
  const [answer, setAnswer] = useState(null)           // null = none, else { text, state: 'streaming' | 'done' | 'error', error }
  const inputRef = useRef(null)
  const abortRef = useRef(null)     // the request in flight, so Zurücksetzen can stop it

  useEffect(() => {
    document.title = UI.title
    inputRef.current?.focus()
    fetch(`${API}/meta`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(String(r.status)))))
      .then(setMeta).catch(() => {})
    fetch(`${API}/health`).then((r) => (r.ok ? r.json() : null)).then((h) => setAnswerOK(Boolean(h && h.answer))).catch(() => {})
  }, [])

  // The year fields start with the real limits (1949 and 2025). A browser steps an EMPTY number field from 0 and
  // clamps it to the minimum, so with only a placeholder both arrow buttons jumped to 1949.
  useEffect(() => {
    if (!meta) return
    setYearFrom((v) => v || String(meta.min_year))
    setYearTo((v) => v || String(meta.max_year))
  }, [meta])
  const yearParam = (v, limit) => (v && Number(v) !== limit ? Number(v) : null)   // the full range needs no filter
  const yearsNarrowed = meta ? yearParam(yearFrom, meta.min_year) !== null || yearParam(yearTo, meta.max_year) !== null : false

  const yearHint = meta ? `${meta.min_year}–${meta.max_year}` : ''

  async function runSearch(e, overrideQ, forceAnswer) {
    if (e) e.preventDefault()
    const q = (overrideQ != null ? overrideQ : query).trim()
    if (!q || loading) return
    if (yearFrom && yearTo && Number(yearFrom) > Number(yearTo)) {
      setError('Das Startjahr liegt nach dem Endjahr. Bitte den Zeitraum anpassen.'); return
    }
    setLoading(true); setError(null)
    abortRef.current?.abort()
    const ac = new AbortController(); abortRef.current = ac
    if ((forceAnswer || withAnswer) && answerOK) { await runAnswer(q, ac); return }
    setAnswer(null)
    try {
      const res = await fetch(`${API}/search`, {
        method: 'POST', signal: ac.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q, top_k: Number(topK) || 10,
          year_start: yearParam(yearFrom, meta?.min_year),
          year_end: yearParam(yearTo, meta?.max_year),
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
      if (err.name === 'AbortError') return
      // fetch() rejects with a bare TypeError ("Failed to fetch") when the server cannot be reached.
      setError(err instanceof TypeError ? 'Der Server ist momentan nicht erreichbar. Bitte gleich erneut versuchen.' : (err.message || 'Unbekannter Fehler'))
      setResults(null)
    } finally {
      if (abortRef.current === ac) setLoading(false)
    }
  }

  // Clears the current question, answer and results, and stops a request that is still running.
  function resetAll() {
    abortRef.current?.abort(); abortRef.current = null
    setQuery(''); setResults(null); setAnswer(null); setError(null)
    setPicked({}); setAsked(''); setLoading(false)
    inputRef.current?.focus()
  }

  // Search plus generated answer. The server streams server-sent events: first the sources (the same
  // cards as an ordinary search, numbered as the answer cites them), then the answer text token by token.
  async function runAnswer(q, ac) {
    const t0 = performance.now()
    setAnswer({ text: '', state: 'streaming', error: null })
    try {
      const res = await fetch(`${API}/answer`, {
        method: 'POST', signal: ac.signal,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: q, top_k: Math.min(Number(topK) || 10, 50),
          year_start: yearParam(yearFrom, meta?.min_year),
          year_end: yearParam(yearTo, meta?.max_year),
        }),
      })
      if (!res.ok) {
        let msg = `Antwort fehlgeschlagen (${res.status})`
        try { const j = await res.json(); if (j && typeof j.detail === 'string') msg = j.detail } catch { /* keep default */ }
        throw new Error(msg)
      }
      if (!(res.headers.get('content-type') || '').includes('text/event-stream')) {
        const j = await res.json()
        setResults(j.results || []); setAsked(q); setPicked({}); setAnswer(null)
        if (j.message) setError(j.message)
        return
      }
      const reader = res.body.getReader()
      const dec = new TextDecoder()
      let buf = ''
      for (;;) {
        const { value, done } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        let cut
        while ((cut = buf.indexOf('\n\n')) >= 0) {
          const line = buf.slice(0, cut).trim(); buf = buf.slice(cut + 2)
          if (!line.startsWith('data:')) continue
          let ev; try { ev = JSON.parse(line.slice(5)) } catch { continue }
          if (ev.type === 'sources') {
            setResults(ev.results); setAsked(q); setPicked({}); setTook(Math.round(performance.now() - t0))
          } else if (ev.type === 'token') {
            setAnswer((a) => ({ ...a, text: a.text + ev.text }))
          } else if (ev.type === 'error') {
            setAnswer((a) => ({ ...a, state: 'error', error: ev.message }))
          } else if (ev.type === 'done') {
            setAnswer((a) => ({ ...a, state: a.state === 'error' ? 'error' : 'done' }))
          }
        }
      }
      setAnswer((a) => (a && a.state === 'streaming' ? { ...a, state: 'done' } : a))
    } catch (err) {
      if (err.name === 'AbortError') return
      setError(err instanceof TypeError ? 'Der Server ist momentan nicht erreichbar. Bitte gleich erneut versuchen.' : (err.message || 'Unbekannter Fehler'))
      setAnswer(null)
    } finally {
      if (abortRef.current === ac) setLoading(false)
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
            {(query || results !== null || answer || error) && (
              <button type="button" className="ghost" onClick={resetAll}
                      title="Frage, Antwort und Treffer löschen">Zurücksetzen</button>
            )}
          </div>
          <div className="controls">
            <span className="ctl">
              <label htmlFor="year-from">Jahr von</label>
              <input id="year-from" type="number" inputMode="numeric" value={yearFrom} placeholder={meta?.min_year ?? ''}
                     min={meta?.min_year} max={meta?.max_year} onChange={(e) => setYearFrom(e.target.value)}
                     onBlur={() => { if (!yearFrom && meta) setYearFrom(String(meta.min_year)) }} />
            </span>
            <span className="ctl">
              <label htmlFor="year-to">bis</label>
              <input id="year-to" type="number" inputMode="numeric" value={yearTo} placeholder={meta?.max_year ?? ''}
                     min={meta?.min_year} max={meta?.max_year} onChange={(e) => setYearTo(e.target.value)}
                     onBlur={() => { if (!yearTo && meta) setYearTo(String(meta.max_year)) }} />
            </span>
            <span className="ctl">
              <label htmlFor="top-k">Treffer</label>
              <select id="top-k" value={topK} onChange={(e) => setTopK(e.target.value)}
                      title={withAnswer && answerOK ? 'So viele Reden fasst das Sprachmodell zusammen; bei vielen Reden sind die Auszüge kürzer' : undefined}>
                {[5, 10, 20, 30, 50].map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            </span>
            {answerOK && (
              <label className="ctl answer-toggle" title="Ein Sprachmodell fasst die ähnlichsten Reden zusammen und belegt die Aussagen mit Quellenverweisen">
                <input type="checkbox" checked={withAnswer} onChange={(e) => setWithAnswer(e.target.checked)} />
                Antwort des Sprachmodells
              </label>
            )}
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
          {results === null && !loading && answerOK && (
            <p className="hint">
              Fragen an das Sprachmodell:{' '}
              {UI.questions.map((qu) => (
                <button type="button" key={qu} className="chip"
                        onClick={() => { setQuery(qu); setWithAnswer(true); runSearch(null, qu, true) }}>{qu}</button>
              ))}
            </p>
          )}
        </form>

        {error && <div className="notice error" role="alert">{error}</div>}

        {answer && (
          <section className="answer" aria-live="polite" aria-busy={answer.state === 'streaming'}>
            <h2>Antwort <span className="tag">maschinell erzeugt</span></h2>
            {answer.text ? <AnswerText text={answer.text} /> : <p className="muted">Das Sprachmodell liest die Reden …</p>}
            {answer.state === 'error' && <p className="notice error">{answer.error || 'Die Antwort konnte nicht erzeugt werden.'}</p>}
            {answer.state !== 'streaming' && (
              <p className="answer-note">
                Das Modell fasst nur die unten stehenden Redeauszüge zusammen und kann sich irren. Aussagen bitte an den Quellen prüfen.
              </p>
            )}
          </section>
        )}

        {results !== null && (
          <section className="results">
            <p className="rmeta" aria-live="polite">
              {results.length > 0
                ? <>{results.length} Reden zu „{asked}“, nach semantischer Ähnlichkeit sortiert · {took} ms</>
                : <>Keine Treffer für „{asked}“{yearsNarrowed ? ' im gewählten Zeitraum' : ''}.</>}
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
              <ResultCard key={`${asked}-${row.date}-${i}`} id={`src-${row.rank}`} numbered={Boolean(answer)} row={row} query={asked}
                          picked={Boolean(picked[i])}
                          onPick={() => setPicked((c) => ({ ...c, [i]: !c[i] }))} />
            ))}
          </section>
        )}
      </main>

      <div className="wrap">
        <aside className="disclaimer" aria-label="Hinweise zur Nutzung">
          <p>
            Unabhängiger Forschungsprototyp, keine offizielle Anwendung des PolMine-Projekts oder des Deutschen
            Bundestags.{answerOK && ' Antworten des Sprachmodells sind maschinell erzeugt und können Fehler enthalten.'}
          </p>
          <details>
            <summary>Hinweise zu Daten, Lizenz, Sprachmodell und Datenschutz</summary>
            <ul>
              <li>
                <strong>Daten und Lizenz.</strong> Die Reden stammen aus GermaParl (PolMine, Blätte et al.) und geben den
                Wortlaut der Plenarprotokolle wieder; maßgeblich bleibt das amtliche Protokoll. Die aufbereitete Fassung
                steht unter CLARIN PUB+BY+NC+SA. Die Lizenz erlaubt nur nichtkommerzielle Nutzung, das gilt auch für
                Texte aus den PDF-, TXT- und CSV-Downloads.
              </li>
              {answerOK && (
                <li>
                  <strong>Sprachmodell.</strong> Die Antworten erzeugt ein offenes Sprachmodell auf einem Server der
                  Universität Bielefeld. Es erhält nur die Frage und Auszüge der angezeigten Reden, belegt seine
                  Aussagen mit Quellennummern und ist als maschinell erzeugt gekennzeichnet. Aussagen bitte an den
                  Quellen prüfen.
                </li>
              )}
              <li>
                <strong>Datenschutz.</strong> Die Seite setzt keine Cookies, speichert nichts im Browser und nutzt kein
                Tracking. Suchtexte und Fragen werden nicht dauerhaft gespeichert und nicht protokolliert; für die
                Nutzungsgrenzen merkt sich der Server die IP-Adresse kurz im Arbeitsspeicher. Der Webserver führt ein
                Zugriffsprotokoll (IP-Adresse, Zeitpunkt, aufgerufene Adresse ohne Suchtext), das nur wenige Tage
                aufbewahrt wird. Daneben speichert der Server pro Tag nur Summen (Seitenaufrufe, Suchen, Antworten und die
                Zahl der verschiedenen Besucheradressen), ohne Adressen. Suchtexte werden auf einem Server der Universität verarbeitet; nur wenn dieser nicht
                erreichbar ist, berechnet der externe Dienst Jina AI den Suchvektor. Allgemeine Angaben:{' '}
                <a href="https://geolab.soz.uni-bielefeld.de/privacy.html" target="_blank" rel="noopener noreferrer">Datenschutzerklärung des GeoLAB</a>{' '}
                (Verantwortlicher: Universität Bielefeld).
              </li>
            </ul>
          </details>
        </aside>
      </div>

      <footer className="foot wrap">
        <p>
          Semantische Suche über rund {meta ? (Number(meta.count) / 1e6).toLocaleString('de-DE', { maximumFractionDigits: 2 }) : '…'} Millionen Reden.
          Angezeigt werden die Originaltreffer, nach Ähnlichkeit sortiert.
          {answerOK ? ' Auf Wunsch fasst ein Sprachmodell die obersten Treffer zusammen; die Quellen stehen immer darunter.' : ' Eine generierte Zusammenfassung gibt es nicht.'}
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
