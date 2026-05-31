"""tests/test_topic_assignments.py — validate topic assignments JSON integrity."""
import json
import pytest
from pathlib import Path
from rag_pipeline.core.paths import Paths


@pytest.fixture(scope="module")
def assignments():
    return json.load(open(Paths.topic_assignments()))


def test_valid_json(assignments):
    assert isinstance(assignments, dict)


def test_required_keys(assignments):
    assert "metadata" in assignments
    assert "results" in assignments


def test_production_model_present(assignments):
    from rag_pipeline.core.paths import Paths
    model = Paths.defaults()["production_model"]
    assert model in assignments["results"], f"{model} missing from results"


def test_ner_entities_field_present(assignments):
    from rag_pipeline.core.paths import Paths
    model = Paths.defaults()["production_model"]
    for a in assignments["results"][model]["assignments"][:20]:
        assert "ner_entities" in a, f"ner_entities missing from doc {a.get('id')}"
        assert isinstance(a["ner_entities"], list)


def test_no_shared_references(assignments):
    """Ensure aliased models have independent data (not shared dict references)."""
    models = list(assignments["results"].keys())
    if len(models) < 2:
        pytest.skip("Need at least 2 models to test")
    a = assignments["results"][models[0]]["assignments"][0]
    b = assignments["results"][models[1]]["assignments"][0]
    assert a is not b, "Models share the same assignment dict reference"
