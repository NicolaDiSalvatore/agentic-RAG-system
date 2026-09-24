"""
Resumable rebuild of the Qdrant corpus collection.

This only adds chunks that are missing from the collection and reuses on-disk embeddings via the content-hash
cache, so an interrupted run can be re-invoked to finish in place. Prints
REBUILD_DONE when every source chunk is indexed.
"""

import argparse
import sys
from pathlib import Path
from src.indexing.build_index import fill_index
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

EXPECTED_COUNT = 36917


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally fill the Qdrant corpus.")
    parser.add_argument("--source", default="data/gutenqa_chunks.parquet")
    parser.add_argument("--expected", type=int, default=EXPECTED_COUNT)
    args = parser.parse_args()

    fill_index(args.source, expected_count=args.expected)
    print(f"REBUILD_DONE points={args.expected} expected={args.expected}")


if __name__ == "__main__":
    main()