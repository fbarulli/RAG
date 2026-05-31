"""tests/test_pydantic_models.py — validate Pydantic model schemas."""
import pytest
from pydantic import ValidationError
from rag_pipeline.core.models.topics import DocNERInfo
from rag_pipeline.ingestion.benchmark_types import QueryResult, MetricSummary, SearchResult


def test_doc_ner_info_defaults():
    d = DocNERInfo()
    assert d.ner_category == "OTHER"
    assert d.ner_primary_entity is None
    assert d.ner_entities == []
    assert d.topic == -1
    assert d.subtopic is None


def test_doc_ner_info_rejects_wrong_type():
    with pytest.raises(ValidationError):
        DocNERInfo(ner_entities="not_a_list")


def test_doc_ner_info_frozen():
    d = DocNERInfo(ner_category="TOOL")
    with pytest.raises(Exception):
        d.ner_category = "OTHER"


def test_query_result_computes_rank():
    r = QueryResult(
        query_id="q1", query_text="test", expected_id="doc2",
        course="test", topic=1, subtopic=None,
        hit_ids=("doc1", "doc2", "doc3"),
        hit_scores=(0.9, 0.8, 0.7),
        latency_ms=5.0, code_integrity_ref=1.0,
        rank=2, hit_at_1=False, hit_at_3=True, hit_at_5=True,
    )
    assert r.rank == 2
    assert r.hit_at_1 is False
    assert r.hit_at_3 is True


def test_query_result_defaults():
    r = QueryResult(
        query_id="q1", query_text="test", expected_id="doc1",
        course="test", topic=None, subtopic=None,
        hit_ids=(), hit_scores=(), latency_ms=5.0, code_integrity_ref=1.0,
    )
    assert r.ner_entities == ()
    assert r.ner_primary_entity is None
    assert r.rank is None
    assert r.hit_at_1 is False


def test_metric_summary_to_dict():
    s = MetricSummary(config_name="entity_boosted", model_name="BAAI/bge-base-en-v1.5")
    d = s.to_dict()
    assert d["config_name"] == "entity_boosted"
    assert d["hit_rate_1"] == 0.0


def test_metric_summary_rejects_wrong_type():
    with pytest.raises(ValidationError):
        MetricSummary(config_name="test", model_name="test", hit_rate_1="not_a_float")
