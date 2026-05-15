# gen/models.py
from typing import TypedDict, Optional, List

class SearchResult(TypedDict):
    """What we get from experiments/results/*.json"""
    k: int
    query: str
    expected_id: str
    found_id: str
    success: bool
    rank: int
    score: float
    found_course: str
    contexts: List[str]
    latency_ms: float

class ContextJudgment(TypedDict):
    """What we get from experiments/judge/*_progress.csv"""
    subset: str
    query: str
    expected_id: str
    found_id: str
    judge_any_yes: str  # 'True'/'False' in CSV
    judge_verdicts: str  # "['YES', 'NO', 'NO']" in CSV
    raw_response: str
    raw_len: str  # number as string in CSV
    verdict_count_raw: str  # number as string in CSV

class GroundTruth(TypedDict):
    """What we fetch from Elasticsearch by expected_id"""
    id: str
    question: str
    answer: str
    course: str
    sort_order: int
    source: str

class GeneratedAnswer(TypedDict):
    """What we produce after generation"""
    subset: str
    query: str
    expected_id: str
    found_id: str
    ground_truth: str
    contexts: List[str]
    generated_answer: str
    answer_len: int

class AnswerJudgment(TypedDict):
    """What we produce after judging"""
    subset: str
    query: str
    expected_id: str
    ground_truth: str
    generated_answer: str
    judge_correct: bool
    judge_reasoning: str
    judge_raw_response: str