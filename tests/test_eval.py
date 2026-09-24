"""Tests for the evaluation pipeline in src/evaluation/run_eval.py."""

import argparse
from unittest.mock import MagicMock, patch
import pytest
from datasets import Dataset
from src.evaluation.run_eval import (
    METRIC_NAMES,
    _parse_args,
    invoke_graph,
    load_gutenqa,
    print_route_metrics,
    run,
)



def test_load_gutenqa_raises_on_non_positive_limit(monkeypatch):
    import src.evaluation.run_eval as run_eval

    monkeypatch.setattr(run_eval, "DATA_PATH", MagicMock(is_file=lambda: True))
    with pytest.raises(ValueError):
        load_gutenqa(0)
    with pytest.raises(ValueError):
        load_gutenqa(-1)


def test_load_gutenqa_raises_when_data_file_missing(monkeypatch):
    import src.evaluation.run_eval as run_eval

    monkeypatch.setattr(run_eval, "DATA_PATH", MagicMock(is_file=lambda: False))
    with pytest.raises(FileNotFoundError):
        load_gutenqa()


def test_load_gutenqa_raises_on_missing_columns(monkeypatch):
    import src.evaluation.run_eval as run_eval

    monkeypatch.setattr(
        run_eval, "DATA_PATH", MagicMock(is_file=lambda: True)
    )
    loaded = Dataset.from_list([{"Question": "q", "Answer": "a"}])
    monkeypatch.setattr(
        run_eval, "load_dataset", MagicMock(return_value=loaded)
    )
    with pytest.raises(ValueError):
        load_gutenqa()


def test_load_gutenqa_returns_mapped_and_filtered_records(monkeypatch):
    import src.evaluation.run_eval as run_eval

    monkeypatch.setattr(run_eval, "DATA_PATH", MagicMock(is_file=lambda: True))
    loaded = Dataset.from_list(
        [
            {
                "Question": "What is France?",
                "Answer": "A country.",
                "Chunk Must Contain": "France is in Europe",
            },
            {
                "Question": "",
                "Answer": "Empty question is dropped",
                "Chunk Must Contain": "no content",
            },
        ]
    )
    monkeypatch.setattr(run_eval, "load_dataset", MagicMock(return_value=loaded))

    result = load_gutenqa(limit=1)
    assert result.column_names == ["question", "ground_truth", "contexts"]
    assert result[0]["question"] == "What is France?"
    assert result[0]["ground_truth"] == "A country."
    assert result[0]["contexts"] == []
    assert len(result) == 1


def test_load_gutenqa_raises_when_no_complete_records(monkeypatch):
    import src.evaluation.run_eval as run_eval

    monkeypatch.setattr(run_eval, "DATA_PATH", MagicMock(is_file=lambda: True))
    loaded = Dataset.from_list(
        [{"Question": "", "Answer": "", "Chunk Must Contain": ""}]
    )
    monkeypatch.setattr(run_eval, "load_dataset", MagicMock(return_value=loaded))
    with pytest.raises(ValueError):
        load_gutenqa()


def test_load_gutenqa_selects_limit(monkeypatch):
    import src.evaluation.run_eval as run_eval

    monkeypatch.setattr(run_eval, "DATA_PATH", MagicMock(is_file=lambda: True))
    loaded = Dataset.from_list(
        [
            {"Question": f"q{i}", "Answer": f"a{i}", "Chunk Must Contain": f"c{i}"}
            for i in range(5)
        ]
    )
    monkeypatch.setattr(run_eval, "load_dataset", MagicMock(return_value=loaded))
    result = load_gutenqa(limit=2)
    assert len(result) == 2
    assert result[0]["question"] == "q0"
    assert result[1]["question"] == "q1"


def test_load_gutenqa_limit_exceeding_length_clamps(monkeypatch):
    import src.evaluation.run_eval as run_eval

    monkeypatch.setattr(run_eval, "DATA_PATH", MagicMock(is_file=lambda: True))
    loaded = Dataset.from_list(
        [
            {"Question": f"q{i}", "Answer": f"a{i}", "Chunk Must Contain": f"c{i}"}
            for i in range(2)
        ]
    )
    monkeypatch.setattr(run_eval, "load_dataset", MagicMock(return_value=loaded))
    result = load_gutenqa(limit=100)
    assert len(result) == 2


def test_invoke_graph_collects_records_in_order():
    graph = MagicMock()

    def fake_invoke(state):
        return {
            "answer": f"answer for {state['question']}",
            "retrieved_chunks": [
                {"text": " context one ", "source": "a.md"},
                {"text": "", "source": "empty.md"},
                {"text": "context two", "source": "b.md"},
                "not a dict",
            ],
            "route": "simple",
        }

    graph.invoke.side_effect = fake_invoke

    examples = Dataset.from_list(
        [
            {"question": "q1", "ground_truth": "g1"},
            {"question": "q2", "ground_truth": "g2"},
        ]
    )

    result = invoke_graph(graph, examples, workers=2)
    assert result.column_names == [
        "question",
        "answer",
        "contexts",
        "ground_truth",
        "route",
    ]
    assert result[0]["question"] == "q1"
    assert result[0]["answer"] == "answer for q1"
    assert result[0]["contexts"] == ["context one", "context two"]
    assert result[0]["ground_truth"] == "g1"
    assert result[0]["route"] == "simple"

    assert graph.invoke.call_count == 2
    assert graph.invoke.call_args_list[0].args[0] == {
        "question": "q1",
        "retry_count": 0,
    }


def test_invoke_graph_with_no_contexts():
    graph = MagicMock()
    graph.invoke.return_value = {"answer": "a", "route": "no_retrieval"}
    examples = Dataset.from_list([{"question": "q1", "ground_truth": "g1"}])
    result = invoke_graph(graph, examples, workers=1)
    assert result[0]["contexts"] == []
    assert result[0]["route"] == "no_retrieval"


def test_print_route_metrics_prints_by_route(capsys):
    import pandas as pd

    df = pd.DataFrame(
        {
            "faithfulness": [0.5, 1.0],
            "answer_relevancy": [0.6, 0.8],
            "context_precision": [0.7, 0.9],
            "context_recall": [0.4, 0.6],
            "route": ["simple", "complex"],
        }
    )
    scores = MagicMock()
    scores.to_pandas.return_value = df

    print_route_metrics(scores)

    out = capsys.readouterr().out
    assert "Aggregate metrics" in out
    assert "faithfulness: 0.7500" in out
    assert "Metrics by observed route" in out
    assert "simple (n=1)" in out
    assert "complex (n=1)" in out


def test_print_route_metrics_raises_when_no_metric_columns():
    import pandas as pd

    scores = MagicMock()
    scores.to_pandas.return_value = pd.DataFrame({"route": ["simple"]})
    with pytest.raises(ValueError):
        print_route_metrics(scores)


def test_print_route_metrics_handles_missing_route(capsys):
    import pandas as pd

    df = pd.DataFrame(
        {
            "faithfulness": [0.5],
            "answer_relevancy": [0.6],
            "context_precision": [0.7],
            "context_recall": [0.4],
        }
    )
    scores = MagicMock()
    scores.to_pandas.return_value = df

    print_route_metrics(scores)

    out = capsys.readouterr().out
    assert "Aggregate metrics" in out
    assert "Route data was not preserved by RAGAS" in out


def test_print_route_metrics_restores_routes_from_input(capsys):
    import pandas as pd

    df = pd.DataFrame(
        {
            "faithfulness": [0.5, 1.0],
            "answer_relevancy": [0.6, 0.8],
            "context_precision": [0.7, 0.9],
            "context_recall": [0.4, 0.6],
        }
    )
    scores = MagicMock()
    scores.to_pandas.return_value = df

    print_route_metrics(scores, routes=["simple", "complex"])

    out = capsys.readouterr().out
    assert "Metrics by observed route" in out
    assert "simple (n=1)" in out
    assert "complex (n=1)" in out
    assert "Route data was not preserved" not in out


def test_print_route_metrics_prefers_existing_route_column(capsys):
    import pandas as pd

    df = pd.DataFrame(
        {
            "faithfulness": [0.5, 1.0],
            "answer_relevancy": [0.6, 0.8],
            "context_precision": [0.7, 0.9],
            "context_recall": [0.4, 0.6],
            "route": ["simple", "simple"],
        }
    )
    scores = MagicMock()
    scores.to_pandas.return_value = df

    print_route_metrics(scores, routes=["complex", "no_retrieval"])

    out = capsys.readouterr().out
    assert "simple (n=2)" in out
    assert "complex (n=1)" not in out


def test_print_route_metrics_ignores_mismatched_routes(capsys):
    import pandas as pd

    df = pd.DataFrame(
        {
            "faithfulness": [0.5],
            "answer_relevancy": [0.6],
            "context_precision": [0.7],
            "context_recall": [0.4],
        }
    )
    scores = MagicMock()
    scores.to_pandas.return_value = df

    print_route_metrics(scores, routes=["simple", "complex"])

    out = capsys.readouterr().out
    assert "Route data was not preserved by RAGAS" in out


def test_print_route_metrics_surfaces_nan_counts(capsys):
    import pandas as pd

    df = pd.DataFrame(
        {
            "faithfulness": [0.5, float("nan")],
            "answer_relevancy": [0.6, 0.8],
            "context_precision": [0.7, 0.9],
            "context_recall": [0.4, 0.6],
            "route": ["simple", "complex"],
        }
    )
    scores = MagicMock()
    scores.to_pandas.return_value = df

    print_route_metrics(scores)

    out = capsys.readouterr().out
    assert "faithfulness: 0.5000 (1/2 rows NaN)" in out
    assert "answer_relevancy: 0.7000" in out
    assert "simple (n=1)" in out
    assert "faithfulness=0.5000" in out
    assert "complex (n=1)" in out
    assert "faithfulness=nan (1/1 NaN)" in out


def test_print_route_metrics_groupby_dropna(monkeypatch, capsys):
    import pandas as pd

    df = pd.DataFrame(
        {
            "faithfulness": [0.5, 1.0],
            "answer_relevancy": [0.6, 0.8],
            "context_precision": [0.7, 0.9],
            "context_recall": [0.4, 0.6],
            "route": ["simple", "simple"],
        }
    )
    scores = MagicMock()
    scores.to_pandas.return_value = df
    print_route_metrics(scores)
    out = capsys.readouterr().out
    assert "simple (n=2)" in out



def test_run_wires_pipeline_components(monkeypatch):
    import src.evaluation.run_eval as run_eval

    examples = Dataset.from_list([{"question": "q", "ground_truth": "g", "contexts": []}])
    records = Dataset.from_list(
        [{"question": "q", "answer": "a", "contexts": ["c"], "ground_truth": "g", "route": "simple"}]
    )
    scores = MagicMock()
    scores.to_pandas.side_effect = NotImplementedError  # run must not print

    monkeypatch.setattr(run_eval, "load_gutenqa", MagicMock(return_value=examples))
    monkeypatch.setattr(run_eval, "invoke_graph", MagicMock(return_value=records))
    monkeypatch.setattr(run_eval, "_build_ragas_llm", MagicMock(return_value="llm"))
    monkeypatch.setattr(run_eval, "_build_ragas_embeddings", MagicMock(return_value="emb"))
    monkeypatch.setattr(run_eval, "print_route_metrics", MagicMock())

    evaluate = MagicMock(return_value=scores)
    with (
        patch("ragas.evaluate", evaluate),
        patch("src.graph.build_graph.build_graph", return_value="graph"),
    ):
        run(limit=1, max_workers=3)

    run_eval.load_gutenqa.assert_called_once_with(1)
    run_eval.invoke_graph.assert_called_once()
    args, kwargs = run_eval.invoke_graph.call_args
    assert args[0] == "graph"
    assert kwargs == {"workers": 3}

    assert evaluate.call_count == 1
    call_args, call_kwargs = evaluate.call_args
    assert call_args[0] is records
    assert call_kwargs["llm"] == "llm"
    assert call_kwargs["embeddings"] == "emb"

    args, kwargs = run_eval.print_route_metrics.call_args
    assert args == (scores,)
    assert kwargs == {"routes": ["simple"]}


def test_parse_args_defaults():
    with patch("sys.argv", ["run_eval"]):
        args = _parse_args()
    assert args.limit is None
    assert args.max_workers == 4


def test_parse_args_accepts_flags():
    with patch("sys.argv", ["run_eval", "--limit", "10", "--max-workers", "8"]):
        args = _parse_args()
    assert args.limit == 10
    assert args.max_workers == 8
