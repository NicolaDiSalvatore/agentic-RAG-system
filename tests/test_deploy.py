"""Static checks that local run and monitoring files agree with the code."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from prometheus_client import REGISTRY

from src.observability import metrics  # noqa: F401 - registers the metrics

ROOT = Path(__file__).parents[1]


def _load(path: str) -> dict:
    return yaml.safe_load((ROOT / path).read_text(encoding="utf-8"))


def test_development_compose_binds_to_loopback():
    services = _load("docker-compose.yml")["services"]
    for name in ("api", "qdrant"):
        assert all(port.startswith("127.0.0.1:") for port in services[name]["ports"]), name


def test_prometheus_scrapes_the_local_api():
    config = _load("deploy/prometheus/prometheus.yml")
    job = config["scrape_configs"][0]
    assert job["static_configs"][0]["targets"] == ["127.0.0.1:8000"]
    assert (ROOT / "deploy" / "prometheus" / config["rule_files"][0]).is_file()


def test_every_alert_uses_a_metric_the_service_exports():
    exported = {
        sample.name
        for family in REGISTRY.collect()
        for sample in family.samples
    } | {family.name for family in REGISTRY.collect()}
    rules = _load("deploy/prometheus/alerts.yml")["groups"][0]["rules"]
    used = {
        name
        for rule in rules
        for name in re.findall(r"\brag_[a-z_]+\b", rule["expr"])
    }
    assert used
    known_bases = {re.sub(r"_(total|created)$", "", name) for name in exported}
    for name in used:
        assert re.sub(r"_total$", "", name) in known_bases, name


@pytest.mark.parametrize("local_only,expected", [("true", "True"), ("false", "False")])
def test_local_files_only_controls_hugging_face_offline_mode(local_only, expected):
    # In a fresh process, because the library reads this once at import time.
    environment = {
        key: value for key, value in os.environ.items() if key != "HF_HUB_OFFLINE"
    }
    environment["EMBEDDING_LOCAL_FILES_ONLY"] = local_only
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import src.indexing.build_index; import huggingface_hub.constants as c; "
            "print(c.HF_HUB_OFFLINE)",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr[-500:]
    assert result.stdout.strip().splitlines()[-1] == expected
