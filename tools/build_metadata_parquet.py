#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd


CORPORA = {
    "DE": {
        "csv": "DE_Master_Chunks.csv",
        "parquet": "de_metadata.parquet",
        "usecols": ["file_id", "date", "speaker_name", "party", "text", "year"],
    },
    "UK": {
        "csv": "Master_Chunks.csv",
        "parquet": "uk_metadata.parquet",
        "usecols": ["date", "speaker", "party", "text"],
    },
}


def main():
    parser = argparse.ArgumentParser(description="Build smaller parquet metadata files for the portable peer bundle.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--corpus", choices=["DE", "UK", "both"], default="both")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    targets = ["DE", "UK"] if args.corpus == "both" else [args.corpus]

    for key in targets:
        config = CORPORA[key]
        src = data_dir / config["csv"]
        dst = data_dir / config["parquet"]
        print(f"[{key}] reading {src}")
        frame = pd.read_csv(src, usecols=config["usecols"], low_memory=False)
        print(f"[{key}] writing {dst}")
        frame.to_parquet(dst, index=False)


if __name__ == "__main__":
    main()
