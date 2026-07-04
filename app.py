import os
import time
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import streamlit as st
# NOTE: sentence_transformers / torch are imported lazily (only for the local-model query
# backend). The default "api" backend calls the Jina embeddings API and needs none of them,
# so an API-only deployment installs just requirements-api.txt (no torch/transformers).


st.set_page_config(page_title="GermaParl Speech Finder", layout="wide")


DATA_DIR = Path(os.environ.get("PORTABLE_PARLIAMENT_DATA_DIR", "/data"))
INDEX_DIR = Path(os.environ.get("PORTABLE_PARLIAMENT_INDEX_DIR", str(DATA_DIR / "indexes")))
CACHE_DIR = Path(os.environ.get("HF_HOME", "/cache/hf"))
EMBED_MODEL = os.environ.get("PORTABLE_PARLIAMENT_EMBED_MODEL", "jinaai/jina-embeddings-v4")
DEFAULT_RESULTS = int(os.environ.get("PORTABLE_PARLIAMENT_RESULTS", "6"))
DEFAULT_CANDIDATE_K = int(os.environ.get("PORTABLE_PARLIAMENT_CANDIDATE_K", "250"))
MAX_CANDIDATE_K = int(os.environ.get("PORTABLE_PARLIAMENT_MAX_CANDIDATE_K", "4000"))
EXACT_FALLBACK_LIMIT = int(os.environ.get("PORTABLE_PARLIAMENT_EXACT_FALLBACK_LIMIT", "250000"))

# Query-embedding backend: "api" (Jina embeddings API — no local model/torch) or "local"
# (SentenceTransformer on CPU). Defaults to "api" when JINA_API_KEY is set, else "local".
# Documents were embedded with jina-embeddings-v4 (task=retrieval, passage, 2048-d, L2-norm),
# so queries use task "retrieval.query"; we L2-normalize the result to match the IP index.
JINA_API_KEY = os.environ.get("JINA_API_KEY", "").strip()
EMBED_BACKEND = os.environ.get("PARLIAMENT_EMBED_BACKEND", "api" if JINA_API_KEY else "local").strip().lower()
JINA_API_URL = os.environ.get("JINA_API_URL", "https://api.jina.ai/v1/embeddings")
JINA_API_MODEL = os.environ.get("JINA_API_MODEL", "jina-embeddings-v4")


@dataclass(frozen=True)
class CorpusConfig:
    key: str
    label: str
    prompt_language: str
    metadata_parquet: str
    metadata_csv: str
    embedding_path: str
    index_path: str
    speaker_col: str
    party_col: str
    date_col: str
    year_col: str | None
    text_col: str
    usecols: tuple[str, ...]
    default_year_range: tuple[int, int]


CORPORA = {
    "DE": CorpusConfig(
        key="DE",
        label="Bundestag (DE)",
        prompt_language="de",
        metadata_parquet="de_metadata.parquet",
        metadata_csv="DE_Master_Chunks.csv",
        embedding_path="embeddings_jina_v4/de_embeddings.npy",
        index_path="indexes/de_ivfpq.faiss",
        speaker_col="speaker_name",
        party_col="party",
        date_col="date",
        year_col="year",
        text_col="text",
        usecols=("file_id", "date", "speaker_name", "party", "text", "year"),
        default_year_range=(1970, 1985),
    ),
    "UK": CorpusConfig(
        key="UK",
        label="House of Commons (UK)",
        prompt_language="en",
        metadata_parquet="uk_metadata.parquet",
        metadata_csv="Master_Chunks.csv",
        embedding_path="embeddings_jina_v4/uk_embeddings.npy",
        index_path="indexes/uk_ivfpq.faiss",
        speaker_col="speaker",
        party_col="party",
        date_col="date",
        year_col=None,
        text_col="text",
        usecols=("date", "speaker", "party", "text"),
        default_year_range=(1997, 2019),
    ),
}


def standardize_metadata(config: CorpusConfig, raw: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["text"] = raw[config.text_col].fillna("").astype(str).str.strip()
    out["speaker"] = raw[config.speaker_col].fillna("").astype(str)
    out["party"] = raw[config.party_col].fillna("").astype(str)
    out["date_raw"] = raw[config.date_col].fillna("").astype(str)
    out["display_date"] = pd.to_datetime(out["date_raw"], errors="coerce")
    if config.year_col and config.year_col in raw.columns:
        out["year"] = pd.to_numeric(raw[config.year_col], errors="coerce")
    else:
        out["year"] = out["display_date"].dt.year

    # Preserve row-for-row alignment with the precomputed embedding arrays.
    out.loc[out["text"] == "", "text"] = "[Text unavailable in portable metadata export]"
    out["year"] = out["year"].fillna(-1).astype(int)
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_metadata(corpus_key: str) -> pd.DataFrame:
    config = CORPORA[corpus_key]
    parquet_path = DATA_DIR / config.metadata_parquet
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        return standardize_metadata(config, df)

    csv_path = DATA_DIR / config.metadata_csv
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {csv_path}")
    df = pd.read_csv(csv_path, usecols=list(config.usecols), low_memory=False)
    return standardize_metadata(config, df)


@st.cache_resource(show_spinner=False)
def load_query_model():
    from sentence_transformers import SentenceTransformer  # lazy: only for the local backend
    os.environ.setdefault("HF_HOME", str(CACHE_DIR))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "hub"))
    return SentenceTransformer(EMBED_MODEL, trust_remote_code=True, device="cpu")


@st.cache_resource(show_spinner=False)
def load_index(corpus_key: str):
    index_path = DATA_DIR / CORPORA[corpus_key].index_path
    if not index_path.exists():
        return None
    index = faiss.read_index(str(index_path))
    if hasattr(index, "nprobe"):
        index.nprobe = min(32, getattr(index, "nlist", 32))
    return index


@st.cache_resource(show_spinner=False)
def load_embeddings(corpus_key: str):
    embedding_path = DATA_DIR / CORPORA[corpus_key].embedding_path
    if not embedding_path.exists():
        raise FileNotFoundError(f"Missing embedding file: {embedding_path}")
    return np.load(embedding_path, mmap_mode="r")


def validate_corpus_alignment(corpus_key: str, df: pd.DataFrame):
    index = load_index(corpus_key)
    embedding_path = DATA_DIR / CORPORA[corpus_key].embedding_path

    if index is None and not embedding_path.exists():
        raise RuntimeError(
            f"{corpus_key} is missing both ANN index and raw embeddings. "
            "At least one retrieval backend must be present."
        )

    if index is not None and index.ntotal != len(df):
        raise RuntimeError(
            f"Metadata/index mismatch for {corpus_key}: {len(df):,} rows vs {index.ntotal:,} indexed vectors"
        )

    if embedding_path.exists():
        embeddings = load_embeddings(corpus_key)
        if embeddings.shape[0] != len(df):
            raise RuntimeError(
                f"Metadata/embedding mismatch for {corpus_key}: {len(df):,} rows vs {embeddings.shape[0]:,} vectors"
            )


def _encode_query_api(query: str) -> np.ndarray:
    import requests
    payload = {"model": JINA_API_MODEL, "task": "retrieval.query", "input": [{"text": query}]}
    resp = requests.post(
        JINA_API_URL, json=payload, timeout=30,
        headers={"Authorization": f"Bearer {JINA_API_KEY}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    vec = np.asarray(resp.json()["data"][0]["embedding"], dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm  # match the L2-normalized docs / inner-product index
    return vec.reshape(1, -1)


def _encode_query_local(query: str) -> np.ndarray:
    model = load_query_model()
    kwargs = {
        "task": "retrieval",
        "prompt_name": "query",
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "batch_size": 1,
    }
    try:
        vector = model.encode(inputs=[query], **kwargs)
    except TypeError:
        try:
            vector = model.encode(sentences=[query], **kwargs)
        except TypeError:
            vector = model.encode([query], convert_to_numpy=True, normalize_embeddings=True, batch_size=1)
    return np.asarray(vector, dtype=np.float32)


def encode_query(query: str) -> np.ndarray:
    if EMBED_BACKEND == "api":
        return _encode_query_api(query)
    return _encode_query_local(query)


def format_source(df: pd.DataFrame, row_idx: int, score: float) -> dict:
    row = df.iloc[row_idx]
    if pd.notnull(row["display_date"]):
        date_label = row["display_date"].strftime("%d.%m.%Y")
    else:
        date_label = str(row["year"])
    return {
        "row_idx": int(row_idx),
        "score": float(score),
        "speaker": row["speaker"],
        "party": row["party"],
        "date": date_label,
        "year": int(row["year"]),
        "text": row["text"],
    }


def exact_search(df: pd.DataFrame, corpus_key: str, query_vec: np.ndarray, year_range: tuple[int, int], top_k: int) -> list[dict]:
    embeddings = load_embeddings(corpus_key)
    mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    positions = np.flatnonzero(mask.to_numpy())
    if len(positions) == 0:
        return []

    batch_size = 50000
    scored: list[tuple[float, int]] = []
    for start in range(0, len(positions), batch_size):
        batch_positions = positions[start:start + batch_size]
        batch_vecs = np.asarray(embeddings[batch_positions], dtype=np.float32)
        scores = batch_vecs @ query_vec[0]
        keep = min(top_k, len(scores))
        local = np.argpartition(scores, -keep)[-keep:]
        for idx in local:
            scored.append((float(scores[idx]), int(batch_positions[idx])))

    scored.sort(key=lambda item: item[0], reverse=True)
    deduped = []
    seen = set()
    for score, row_idx in scored:
        if row_idx in seen:
            continue
        seen.add(row_idx)
        deduped.append(format_source(df, row_idx, score))
        if len(deduped) >= top_k:
            break
    return deduped


def ann_search(df: pd.DataFrame, corpus_key: str, query_vec: np.ndarray, year_range: tuple[int, int], top_k: int, candidate_k: int) -> list[dict]:
    index = load_index(corpus_key)
    embedding_path = DATA_DIR / CORPORA[corpus_key].embedding_path
    has_embeddings = embedding_path.exists()
    if index is None:
        if not has_embeddings:
            raise RuntimeError(
                f"{corpus_key} retrieval needs either {CORPORA[corpus_key].index_path} "
                f"or {CORPORA[corpus_key].embedding_path}."
            )
        return exact_search(df, corpus_key, query_vec, year_range, top_k)

    search_k = max(candidate_k, top_k * 8)
    seen = set()
    collected = []
    while search_k <= MAX_CANDIDATE_K and len(collected) < top_k:
        distances, ids = index.search(query_vec, search_k)
        for score, row_idx in zip(distances[0], ids[0]):
            row_idx = int(row_idx)
            if row_idx < 0 or row_idx in seen:
                continue
            seen.add(row_idx)
            year = int(df.iloc[row_idx]["year"])
            if year < year_range[0] or year > year_range[1]:
                continue
            collected.append(format_source(df, row_idx, float(score)))
            if len(collected) >= top_k:
                break
        search_k *= 2

    if len(collected) >= top_k:
        return collected[:top_k]

    year_mask = (df["year"] >= year_range[0]) & (df["year"] <= year_range[1])
    if has_embeddings and int(year_mask.sum()) <= EXACT_FALLBACK_LIMIT:
        return exact_search(df, corpus_key, query_vec, year_range, top_k)
    return collected


def result_summary(sources: list[dict], year_range: tuple[int, int], language: str) -> str:
    """Neutral header for the ranked results. There is deliberately no model-generated
    summary: users read the similarity-ranked speeches themselves, which also keeps the
    app dependency-light (no local LLM / Ollama) for external hosting."""
    if not sources:
        if language == "de":
            return "Keine passenden Reden im gewaehlten Zeitraum gefunden."
        return "No matching speeches were found in the selected time range."
    if language == "de":
        return (
            f"{len(sources)} Reden zu Ihrer Anfrage, nach semantischer Aehnlichkeit sortiert "
            f"({year_range[0]}-{year_range[1]}). Die Originaltexte stehen unten."
        )
    return (
        f"Top {len(sources)} speeches for your query, ranked by semantic similarity "
        f"({year_range[0]}-{year_range[1]}). Read the original passages below."
    )


def init_chat_state(corpus_key: str):
    if "portable_active_corpus" not in st.session_state:
        st.session_state.portable_active_corpus = corpus_key
    if "portable_messages" not in st.session_state:
        st.session_state.portable_messages = []
    if st.session_state.portable_active_corpus != corpus_key:
        st.session_state.portable_active_corpus = corpus_key
        st.session_state.portable_messages = []


def render_source_cards(sources: list[dict], corpus_label: str):
    st.subheader(f"Sources from {corpus_label}")
    for rank, source in enumerate(sources, start=1):
        with st.container(border=True):
            st.markdown(
                f"**{rank}. {source['speaker']}** ({source['party']}) — {source['date']} "
                f"`Score {source['score']:.3f}`"
            )
            st.write(source["text"][:650] + ("..." if len(source["text"]) > 650 else ""))
            with st.expander("Show full retrieved chunk"):
                st.write(source["text"])


_available = [
    k for k in CORPORA
    if (DATA_DIR / CORPORA[k].index_path).exists() or (DATA_DIR / CORPORA[k].embedding_path).exists()
]
with st.sidebar:
    st.header("GermaParl Finder")
    if not _available:
        st.error("No corpus index found. Add data/indexes/*.faiss (see README).")
        st.stop()
    corpus_key = st.selectbox("Corpus", options=_available, format_func=lambda key: CORPORA[key].label)

init_chat_state(corpus_key)

df = load_metadata(corpus_key)
validate_corpus_alignment(corpus_key, df)
config = CORPORA[corpus_key]
min_year = int(df["year"].min())
max_year = int(df["year"].max())
valid_years = df.loc[df["year"] > 0, "year"]
if not valid_years.empty:
    min_year = int(valid_years.min())
    max_year = int(valid_years.max())

with st.sidebar:
    year_range = st.slider(
        "Year range",
        min_value=min_year,
        max_value=max_year,
        value=(
            max(min_year, config.default_year_range[0]),
            min(max_year, config.default_year_range[1]),
        ),
    )
    result_count = st.selectbox("Retrieved speeches", options=[3, 5, 6, 8, 10, 12], index=2)
    candidate_k = st.slider("ANN candidate pool", min_value=100, max_value=2000, value=DEFAULT_CANDIDATE_K, step=100)
    if st.button("New chat", use_container_width=True):
        st.session_state.portable_messages = []
        st.rerun()

    index_path = DATA_DIR / config.index_path
    metadata_path = DATA_DIR / config.metadata_parquet if (DATA_DIR / config.metadata_parquet).exists() else DATA_DIR / config.metadata_csv
    st.caption(f"Metadata: {metadata_path.name}")
    st.caption(f"ANN index: {'available' if index_path.exists() else 'missing'}")
    st.caption(
        f"Raw embeddings: {'available' if (DATA_DIR / config.embedding_path).exists() else 'not bundled'}"
    )
    st.caption(f"Query embedding: {'Jina v4 API' if EMBED_BACKEND == 'api' else EMBED_MODEL + ' (CPU)'}")
    st.caption("Retrieval-only: ranked speeches, no model-generated summary.")


st.title("Portable Parliamentary RAG")
st.caption("Semantic search over precomputed Jina v4 parliamentary embeddings — speeches ranked by similarity, no model-generated summary.")

if not index_path.exists() and not (DATA_DIR / config.embedding_path).exists():
    st.error(
        "No retrieval backend is available. Add a FAISS index or the raw embedding file for this corpus."
    )
    st.stop()

if index_path.exists() and not (DATA_DIR / config.embedding_path).exists():
    st.info(
        "Running in index-only mode. That is the recommended portable setup; "
        "exact-search fallback is disabled to keep the bundle smaller."
    )

if st.session_state.portable_messages:
    latest = st.session_state.portable_messages[-1]
    st.caption(f"Last query searched {config.label} in {latest['year_range'][0]}-{latest['year_range'][1]}.")

for message in st.session_state.portable_messages:
    st.chat_message("user").markdown(message["query"])
    with st.chat_message("assistant"):
        st.markdown(message["answer"])
        render_source_cards(message["sources"], config.label)

placeholder = "Ask a question about parliamentary debates..."
if config.prompt_language == "de":
    placeholder = "Frage zur Parlamentsgeschichte..."

if query := st.chat_input(placeholder):
    query_vec = encode_query(query)
    active_year_range = tuple(year_range)
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Searching the corpus..."):
            sources = ann_search(df, corpus_key, query_vec, active_year_range, result_count, candidate_k)
            answer = result_summary(sources, active_year_range, config.prompt_language)
            if not sources:
                st.warning(answer)
            else:
                st.markdown(answer)
                render_source_cards(sources, config.label)
    st.session_state.portable_messages.append(
        {
            "query": query,
            "answer": answer,
            "sources": sources,
            "year_range": active_year_range,
        }
    )
