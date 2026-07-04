#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import faiss
import numpy as np


CORPORA = {
    "DE": {
        "embedding": "embeddings_jina_v4/de_embeddings.npy",
        "index": "indexes/de_ivfpq.faiss",
        "nlist": 2048,
    },
    "UK": {
        "embedding": "embeddings_jina_v4/uk_embeddings.npy",
        "index": "indexes/uk_ivfpq.faiss",
        "nlist": 4096,
    },
}


def build_index(
    embedding_path: Path,
    output_path: Path,
    nlist: int,
    sample_size: int,
    batch_size: int,
    m: int,
    nbits: int,
):
    embeddings = np.load(embedding_path, mmap_mode="r")
    dim = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, nbits, faiss.METRIC_INNER_PRODUCT)

    rng = np.random.default_rng(42)
    train_size = min(sample_size, embeddings.shape[0])
    train_ids = np.sort(rng.choice(embeddings.shape[0], size=train_size, replace=False))
    train_vectors = np.asarray(embeddings[train_ids], dtype=np.float32)
    print(
        f"Training on {train_vectors.shape[0]:,} vectors from {embedding_path.name} "
        f"(nlist={nlist}, m={m}, nbits={nbits})",
        flush=True,
    )
    index.train(train_vectors)

    print(f"Adding {embeddings.shape[0]:,} vectors to {output_path.name}", flush=True)
    for start in range(0, embeddings.shape[0], batch_size):
        batch = np.asarray(embeddings[start:start + batch_size], dtype=np.float32)
        index.add(batch)
        if start == 0 or start % (batch_size * 10) == 0:
            print(f"  added {min(start + batch.shape[0], embeddings.shape[0]):,} / {embeddings.shape[0]:,}", flush=True)

    if hasattr(index, "nprobe"):
        index.nprobe = min(32, nlist)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(output_path))
    print(f"Wrote {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build FAISS IVFPQ indexes for the portable peer bundle.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--corpus", choices=["DE", "UK", "both"], default="both")
    parser.add_argument("--sample-size", type=int, default=160000)
    parser.add_argument("--batch-size", type=int, default=50000)
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 1) - 1))
    args = parser.parse_args()

    faiss.omp_set_num_threads(args.threads)
    data_dir = Path(args.data_dir)
    targets = ["DE", "UK"] if args.corpus == "both" else [args.corpus]

    for key in targets:
        config = CORPORA[key]
        build_index(
            data_dir / config["embedding"],
            data_dir / config["index"],
            nlist=config["nlist"],
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            m=args.m,
            nbits=args.nbits,
        )


if __name__ == "__main__":
    main()
