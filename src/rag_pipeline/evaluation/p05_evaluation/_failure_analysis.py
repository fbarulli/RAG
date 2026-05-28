"""
Public Functions for Retrieval Failure and Error Stratification Analysis:

def main() -> None:
    Executes a complete profiling analysis of retrieval failures across query types and NER semantic categories, surfacing the highest-impact false positives.
    I/O: None -> None
"""
import json
from collections import Counter, defaultdict
from pathlib import Path
from rag_pipeline.core.paths import Paths
from rag_pipeline.logging import get_logger
logger = get_logger(__name__)


def main() -> None:
    _defaults = Paths.defaults()
    results = json.load(open(Paths.experiments_dir() / 'benchmark_query_results.json'))
    ner_map = {a['id']: a for a in json.load(open(Paths.topic_assignments()))['results'][_defaults['production_model']]['assignments']}
    failures = [r for r in results if not r['hit']]
    passes = [r for r in results if r['hit']]
    print(f'Total: {len(results)} | Pass: {len(passes)} | Fail: {len(failures)}\n')
    print('=== BY QUERY TYPE ===')
    type_total = Counter((r['query_type'] for r in results))
    type_fail = Counter((r['query_type'] for r in failures))
    for qt, total in type_total.most_common():
        fail = type_fail.get(qt, 0)
        print(f'  {qt:20s} fail={fail:3d}/{total:3d} ({fail / total:.1%})')
    print('\n=== BY NER CATEGORY ===')
    cat_total = defaultdict(int)
    cat_fail = defaultdict(int)
    for r in results:
        doc = ner_map.get(r['expected_id'], {})
        cat = doc.get('ner_category', 'UNKNOWN')
        cat_total[cat] += 1
        if not r['hit']:
            cat_fail[cat] += 1
    for cat, total in sorted(cat_total.items(), key=lambda x: -x[1]):
        fail = cat_fail.get(cat, 0)
        print(f'  {cat:10s} fail={fail:3d}/{total:3d} ({fail / total:.1%})')
    print('\n=== SAMPLE FAILURES (first 10) ===')
    for r in failures[:10]:
        doc = ner_map.get(r['expected_id'], {})
        print(f"  [{r['query_type']:18s}] [{doc.get('ner_category', '?'):8s}] {r['query_text'][:70]}")


if __name__ == '__main__':
    main()