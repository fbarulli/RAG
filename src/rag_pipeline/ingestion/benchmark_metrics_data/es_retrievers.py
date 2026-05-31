"""Pure BM25 retrieval via Elasticsearch."""
import time
from typing import Optional, TYPE_CHECKING
from ..benchmark_types import SearchResult

if TYPE_CHECKING:
    from elasticsearch import Elasticsearch

def run_es_retrieval(es: 'Elasticsearch', index: str, query_text: str, course_filter: Optional[str], config: dict, top_k: int) -> SearchResult:
    start = time.perf_counter()
    boost_q = config.get('boost_question', 1.0)
    boost_t = config.get('boost_text', 1.0)
    
    if boost_q > 0 and boost_t > 0:
        must_clause = [{'multi_match': {'query': query_text, 'fields': [f'question^{boost_q}', f'answer^{boost_t}']}}]
    elif boost_q > 0:
        must_clause = [{'match': {'question': {'query': query_text, 'boost': boost_q}}}]
    else:
        must_clause = [{'match': {'answer': {'query': query_text, 'boost': boost_t}}}]
        
    body = {
        'query': {
            'bool': {
                'must': must_clause, 
                'filter': [{'term': {'course': course_filter}}] if course_filter else []
            }
        }, 
        'size': top_k
    }
    
    resp = es.search(index=index, body=body)
    latency_ms = (time.perf_counter() - start) * 1000
    hits = resp['hits']['hits']
    
    return SearchResult(
        hit_ids=tuple(str(h['_source'].get('es_id', '')) for h in hits),  # coerce legacy int es_ids
        hit_scores=tuple(float(h['_score']) for h in hits),
        hit_courses=tuple(h['_source'].get('course', '') for h in hits),
        top_answer=hits[0]['_source'].get('answer') if hits else None,
        latency_ms=latency_ms,
        hit_answers=tuple()
    )

