import argparse
from src.indexing.build_index import build_index

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="File or directory to ingest")
    args = parser.parse_args()

    index = build_index(args.path)
    print(f"Index built and persisted to Qdrant from {args.path}")
