"""Measure retrieval quality and choose EVIDENCE_MIN_DENSE_SCORE from real data.

GutenQA labels each question with the passage that answers it, so this reports
how often each retriever finds that passage, then shows how well the best dense
score separates corpus questions from questions the corpus cannot answer.

Makes no LLM calls. Stop the API first when using embedded Qdrant, because only
one process can open it.

    python -m scripts.calibrate_evidence --sample 300
"""

import argparse
import json
import random
import re
import statistics
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, settings
from src.graph.nodes import _get_retriever, _rerank
from src.indexing.build_index import active_index_version

# Topics the Gutenberg corpus cannot answer. Some are worded like literature
# questions on purpose, because those are the hard cases for a threshold.
OFF_TOPIC_QUESTIONS = [
    "What is the boiling point of liquid nitrogen?",
    "How do I configure a Kubernetes ingress controller?",
    "What is the capital gains tax rate in Germany?",
    "Who won the 2018 FIFA World Cup?",
    "How does quantum entanglement affect semiconductor lithography?",
    "What is the recommended dose of ibuprofen for adults?",
    "How do neural networks use backpropagation?",
    "What is the current population of Tokyo?",
    "How do I reset the password on a Wi-Fi router?",
    "What is the difference between HTTP and HTTPS?",
    "Which smartphone has the best camera this year?",
    "How do mRNA vaccines train the immune system?",
    "What is the airspeed of a Boeing 747 at cruising altitude?",
    "How do I write a recursive function in Python?",
    "What does a blockchain consensus algorithm do?",
    "How much does a solar panel installation cost?",
    "What are the side effects of statins?",
    "Who founded the company that makes the iPhone?",
    "How do electric car batteries degrade over time?",
    "What is the half-life of carbon-14?",
    "What is the plot of the movie Inception?",
    "Who plays the lead role in the television series Breaking Bad?",
    "What did the wizard Gandalf say at the bridge of Khazad-dum?",
    "How does Harry Potter defeat Voldemort?",
    "What is the name of Luke Skywalker's father?",
    "How do you make sourdough starter from scratch?",
    "What is the offside rule in football?",
    "How do interest rates affect mortgage payments?",
    "What is the tallest building in Dubai?",
    "How do I file a tax return as a freelancer?",
    "What programming language is used for Android apps?",
    "What is the speed limit on German motorways?",
    "How does a nuclear fusion reactor confine plasma?",
    "What is the chemical formula of caffeine?",
    "Which planet has the most confirmed moons?",
    "How do I train a puppy to stop biting?",
    "What are the symptoms of type 2 diabetes?",
    "How many gigabytes does a 4K movie use per hour?",
    "What is the role of the European Central Bank?",
    "How does GPS calculate a position from satellites?",
]

THRESHOLDS = [round(0.20 + 0.05 * step, 2) for step in range(11)]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _first_hit_rank(hits: list, passage: str) -> int | None:
    """Return the 1-based rank of the first hit containing the labeled passage."""
    for rank, hit in enumerate(hits, start=1):
        if passage in _normalize(hit.node.get_content()):
            return rank
    return None


def _summarize(ranks: list[int | None], total: int) -> dict[str, float]:
    found = [rank for rank in ranks if rank is not None]
    return {
        "hit_at_1": round(sum(rank <= 1 for rank in found) / total, 3),
        "hit_at_3": round(sum(rank <= 3 for rank in found) / total, 3),
        "hit_at_8": round(sum(rank <= 8 for rank in found) / total, 3),
        "mrr": round(sum(1 / rank for rank in found) / total, 3),
    }


def _percentiles(scores: list[float]) -> dict[str, float]:
    cuts = statistics.quantiles(scores, n=20)
    return {
        "p5": round(cuts[0], 3),
        "p25": round(cuts[4], 3),
        "median": round(cuts[9], 3),
        "p75": round(cuts[14], 3),
        "p95": round(cuts[18], 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=PROJECT_ROOT / ".cache" / "eval" / "evidence.json")
    args = parser.parse_args()

    questions = pd.read_parquet(PROJECT_ROOT / "data" / "questions.parquet")
    rows = questions.to_dict("records")
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.sample]

    retriever = _get_retriever(settings.auth_tenant_id, ["public"])
    ranks: dict[str, list[int | None]] = {
        "served": [],
        "hybrid_rrf": [],
        "dense_only": [],
        "bm25_only": [],
    }
    if settings.enable_reranker:
        ranks["hybrid_reranked"] = []
    on_topic: list[float] = []
    on_topic_found: list[float] = []
    on_topic_missed: list[float] = []

    for index, row in enumerate(rows, start=1):
        question = str(row["Question"])
        passage = _normalize(str(row["Chunk Must Contain"]))
        served = retriever.retrieve(question)
        rank = _first_hit_rank(served, passage)
        ranks["served"].append(rank)
        # Named explicitly, so this stays a real hybrid baseline on any profile.
        hybrid = retriever.retrieve(question, ranking="rrf")
        ranks["hybrid_rrf"].append(_first_hit_rank(hybrid, passage))
        if settings.enable_reranker:
            ranks["hybrid_reranked"].append(_first_hit_rank(_rerank(hybrid, question), passage))
        ranks["dense_only"].append(_first_hit_rank(retriever._dense.retrieve(question), passage))
        ranks["bm25_only"].append(_first_hit_rank(retriever._sparse.retrieve(question), passage))
        score = served[0].top_dense_score if served else None
        if score is not None:
            on_topic.append(score)
            (on_topic_found if rank is not None else on_topic_missed).append(score)
        if index % 50 == 0:
            print(f"{index}/{len(rows)} questions", flush=True)

    off_topic = []
    for question in OFF_TOPIC_QUESTIONS:
        hits = retriever.retrieve(question)
        if hits and hits[0].top_dense_score is not None:
            off_topic.append(hits[0].top_dense_score)

    def share_at_or_above(scores: list[float], threshold: float) -> float:
        return round(sum(score >= threshold for score in scores) / len(scores), 3)

    report = {
        "index_version": active_index_version(),
        "embedding_model": settings.embedding_model,
        "sample": len(rows),
        "off_topic_sample": len(off_topic),
        "seed": args.seed,
        "top_k": settings.top_k_dense,
        "served_ranking": settings.retrieval_ranking,
        "retrieval": {name: _summarize(found, len(rows)) for name, found in ranks.items()},
        "top_dense_score": {
            "on_topic": _percentiles(on_topic),
            "on_topic_passage_found": _percentiles(on_topic_found),
            "on_topic_passage_missed": _percentiles(on_topic_missed),
            "off_topic": _percentiles(off_topic),
        },
        "thresholds": [
            {
                "threshold": threshold,
                "on_topic_covered": share_at_or_above(on_topic, threshold),
                "off_topic_covered": share_at_or_above(off_topic, threshold),
            }
            for threshold in THRESHOLDS
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
