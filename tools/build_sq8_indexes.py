#!/usr/bin/env python3
"""Build flat 8-bit scalar-quantized (SQ8) FAISS indexes for the finder.

SQ8 is the recommended index: on the 2048-d Jina v4 vectors it is near-exact
(recall@10 ~1.0 vs a brute-force baseline) at ~2 GB per ~1M speeches, and CPU-fast
(~150 ms/query — negligible next to the query-embedding API call).

NOTE: the alternative build_ivfpq_indexes.py is ~30x smaller (~60 MB) but its product
quantization is far too lossy on these high-dim vectors (measured recall@10 ~0.32 even at
full nprobe), so prefer SQ8 unless disk/RAM is extremely tight.
"""
import argparse
import os
from pathlib import Path

import faiss
import numpy as np

CORPORA = {
    "DE": {"embedding": "embeddings_jina_v4/de_embeddings.npy", "index": "indexes/de_sq8.faiss"},
    "UK": {"embedding": "embeddings_jina_v4/uk_embeddings.npy", "index": "indexes/uk_sq8.faiss"},
}


def build(embedding_path: Path, output_path: Path, train_size: int, batch_size: int):
    emb = np.load(embedding_path, mmap_mode="r")
    n, dim = emb.shape
    index = faiss.IndexScalarQuantizer(dim, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    rng = np.random.default_rng(42)
    tr = np.sort(rng.choice(n, size=min(train_size, n), replace=False))
    print(f"Training SQ8 on {len(tr):,} vectors from {embedding_path.name}", flush=True)
    index.train(np.ascontiguousarray(emb[tr], dtype=np.float32))
    for s in range(0, n, batch_size):
        index.add(np.ascontiguousarray(emb[s:s + batch_size], dtype=np.float32))
        print(f"  added {min(s + batch_size, n):,} / {n:,}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(f"Wrote {output_path} (ntotal={index.ntotal:,}, {os.path.getsize(output_path)/1e9:.2f} GB)", flush=True)


def main():
    p = argparse.ArgumentParser(description="Build flat SQ8 FAISS indexes (near-exact, ~2 GB/1M vectors).")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--corpus", choices=["DE", "UK", "both"], default="DE")
    p.add_argument("--train-size", type=int, default=200000)
    p.add_argument("--batch-size", type=int, default=100000)
    p.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    a = p.parse_args()
    faiss.omp_set_num_threads(a.threads)
    data_dir = Path(a.data_dir)
    for key in (["DE", "UK"] if a.corpus == "both" else [a.corpus]):
        c = CORPORA[key]
        build(data_dir / c["embedding"], data_dir / c["index"], a.train_size, a.batch_size)


if __name__ == "__main__":
    main()
