"""Evaluate the LangGraph RAG pipeline against the GutenQA data set.

``questions.parquet`` supplies questions, reference answers, and the
supporting passage for each answer. The graph still retrieves its own
contexts; the supporting passage is used only to validate data integrity.

Run all examples with ``python -m src.evaluation.run_eval`` or use
``--limit 25`` for a small smoke test.
"""

from __future__ import annotations
import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import truststore
from datasets import Dataset, load_dataset
# Use Windows' certificate store for the Groq and Hugging Face clients, and
# avoid permission failures in a shared user-level Hugging Face cache.
PROJECT_ROOT = Path(__file__).parents[2]
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / ".cache" / "huggingface"))
truststore.inject_into_ssl()



DATA_PATH = PROJECT_ROOT / "data" / "questions.parquet"
CACHE_DIR = PROJECT_ROOT / ".cache" / "datasets"
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def load_gutenqa(limit: int | None = None) -> Dataset:
    """Load validated GutenQA records in the RAGAS input format."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be a positive integer or None")
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"GutenQA questions file not found: {DATA_PATH}")

    dataset = load_dataset("parquet", data_files=str(DATA_PATH), split="train", cache_dir=str(CACHE_DIR))
    if limit is not None:
        dataset = dataset.select(range(min(limit, len(dataset))))

    required_columns = {"Question", "Answer", "Chunk Must Contain"}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(f"GutenQA is missing required columns: {sorted(missing_columns)}")

    records = []
    for row in dataset:
        question = str(row["Question"] or "").strip()
        answer = str(row["Answer"] or "").strip()
        reference_context = str(row["Chunk Must Contain"] or "").strip()
        if question and answer and reference_context:
            records.append({"question": question, "ground_truth": answer, "contexts": []})

    if not records:
        raise ValueError("GutenQA contains no complete question/answer/context records")
    return Dataset.from_list(records)


def invoke_graph(graph: Any, examples: Dataset, workers: int = 4) -> Dataset:
    """Invoke the graph for each question concurrently and capture route/context."""
    questions = list(examples["question"])
    ground_truths = list(examples["ground_truth"])
    total = len(questions)
    results: list[dict[str, Any] | None] = [None] * total

    def run_one(index: int) -> None:
        state = graph.invoke({"question": questions[index], "retry_count": 0})
        chunks = state.get("retrieved_chunks") or []
        contexts = [
            str(chunk.get("text") or "").strip()
            for chunk in chunks
            if isinstance(chunk, dict) and str(chunk.get("text") or "").strip()
        ]
        results[index] = {
            "question": questions[index],
            "answer": str(state.get("answer") or "").strip(),
            "contexts": contexts,
            "ground_truth": ground_truths[index],
            "route": str(state.get("route") or "unknown"),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, index): index for index in range(total)}
        for done, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            future.result()
            print(f"Ran {done}/{total}: {results[index]['route']}")

    records = [record or {} for record in results]
    return Dataset.from_list(records)


def _build_ragas_llm() -> Any:
    """Use a dedicated model as the RAGAS judge.

    Kept separate from the production model so scoring is not affected by
    generation prompt drift, and so a JSON-reliable model can be chosen for
    the statement/question extraction prompts.
    """
    from ragas.llms import LangchainLLMWrapper

    from src.llm.langchain_llm import get_judge_chat_model

    return LangchainLLMWrapper(get_judge_chat_model())


def _build_ragas_embeddings() -> Any:
    """Reuse the retrieval embedding model for RAGAS metrics."""
    from ragas.cache import DiskCacheBackend
    from ragas.embeddings import LlamaIndexEmbeddingsWrapper

    from src.config import get_embed_model

    # Wrap the cached singleton (same bge-small-en-v1.5 used by retrieval)
    # so no second ~130 MB model is loaded, and let RAGAS's DiskCacheBackend
    # dedupe identical embed_query/embed_documents calls across rows and runs.
    return LlamaIndexEmbeddingsWrapper(
        get_embed_model(),
        cache=DiskCacheBackend(str(PROJECT_ROOT / ".cache" / "ragas_embeddings")),
    )


def print_route_metrics(scores: Any, routes: list[str] | None = None) -> None:
    """Print aggregate RAGAS scores and means grouped by observed route.

    RAGAS 0.3.x drops unknown columns (like ``route``) when it re-validates
    the input rows, so the caller hands in the routes captured before
    evaluation and they are restored here positionally: RAGAS aligns its
    score rows to the input rows in the same order.
    """
    frame = scores.to_pandas()
    available_metrics = [name for name in METRIC_NAMES if name in frame.columns]
    if not available_metrics:
        raise ValueError("RAGAS returned no requested metric columns")

    print("\nAggregate metrics")
    for metric in available_metrics:
        missing = int(frame[metric].isna().sum())
        mean = frame[metric].mean()
        if missing:
            print(f"  {metric}: {mean:.4f} ({missing}/{len(frame)} rows NaN)")
        else:
            print(f"  {metric}: {mean:.4f}")

    if "route" not in frame.columns and routes is not None and len(routes) == len(frame):
        frame["route"] = routes

    if "route" not in frame.columns:
        print("\nRoute data was not preserved by RAGAS; route breakdown unavailable.")
        return
    print("\nMetrics by observed route")
    for route, group in frame.groupby("route", dropna=False):
        entries = []
        for metric in available_metrics:
            missing = int(group[metric].isna().sum())
            mean = group[metric].mean()
            if missing:
                entries.append(f"{metric}={mean:.4f} ({missing}/{len(group)} NaN)")
            else:
                entries.append(f"{metric}={mean:.4f}")
        print(f"  {route} (n={len(group)}): " + ", ".join(entries))


def run(limit: int | None = None, max_workers: int = 4) -> None:
    """Evaluate GutenQA examples, or the first ``limit`` if supplied."""
    # Delay importing the graph so data helpers do not load index dependencies.
    from ragas import RunConfig, evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    from src.graph.build_graph import build_graph

    examples = load_gutenqa(limit)
    metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
    # Groq's chat API rejects n>1 generations per request ("n : number must
    # be at most 1"), but answer_relevancy defaults to generating
    # strictness=3 questions per answer. Cap it so the metric can score.
    answer_relevancy.strictness = 1
    graph = build_graph()
    # Build the index/retriever once, single-threaded, before workers start.
    # Qdrant local mode locks its storage folder, so opening it from many
    # threads at once fails with AlreadyLocked; warming up removes the race.
    from src.graph.nodes import _get_retriever

    _get_retriever()
    records = invoke_graph(graph, examples, workers=max_workers)
    routes = [str(row["route"]) for row in records]
    scores = evaluate(
        records,
        metrics=metrics,
        llm=_build_ragas_llm(),
        embeddings=_build_ragas_embeddings(),
        run_config=RunConfig(max_workers=max_workers),
        show_progress=True,
    )
    print_route_metrics(scores, routes=routes)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the RAG graph on GutenQA.")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N questions.")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Concurrent questions during generation and RAGAS metric calls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(args.limit, max_workers=args.max_workers)
