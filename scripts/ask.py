import argparse
import os
import sys
from pathlib import Path
import truststore
from src.graph.build_graph import build_graph


PROJECT_ROOT = Path(__file__).parents[1]
os.environ.setdefault(
    "HF_HOME",
    str(PROJECT_ROOT / ".cache" / "huggingface"),
)

truststore.inject_into_ssl()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Ask one question to the RAG graph"
    )
    parser.add_argument("question")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print sub-questions, per-attempt retrieval, and per-sub-question grade verdicts",
    )
    args = parser.parse_args()

    app = build_graph()

    path = []
    result = {"question": args.question, "retry_count": 0}
    for step in app.stream(
        {"question": args.question, "retry_count": 0},
        stream_mode="updates",
    ):
        for node_name, update in step.items():
            path.append(node_name)
            result.update(update)
            if args.debug and node_name == "decompose":
                print("\nSUB-QUESTIONS:")
                for sq in update.get("sub_questions", []):
                    print("  -", sq)
            if args.debug and node_name == "retrieve":
                print(f"\nRETRIEVE (attempt retry_count={result.get('retry_count', 0)}):")
                for i, chunk in enumerate(update.get("retrieved_chunks", []), 1):
                    print(f"  [{i}] query={chunk.get('query', '?')!r}")
                    print(f"      source={chunk.get('source', '')}")
                    print(f"      text={str(chunk.get('text', ''))[:120]!r}")
            if args.debug and node_name == "grade":
                details = update.get("grade_details", [])
                print(f"\nGRADE: {'sufficient' if update.get('context_sufficient') else 'insufficient'}")
                for d in details:
                    print(f"  covered={d.get('covered')} | {d.get('sub_question')}")

    print("\nPATH:", " -> ".join(path))

    print("\nROUTE:", result.get("route"))
    print("ANSWER:", result.get("answer"))
    print("SOURCES:")

    for i, chunk in enumerate(result.get("retrieved_chunks", []), 1):
        print(f"\nSOURCE {i}:")
        print(str(chunk.get("text", ""))[:1000])
