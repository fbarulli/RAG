import json
import time
import logging
import traceback
import os
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from src.search import CourseRAGManager
from src.config_manager import load_full_config
from eval.eval_set import get_eval_set_from_es


class BenchmarkRunner:
    def __init__(self, config_name: str, batch_size: int = 50):
        self.config_name = config_name
        self.settings = load_full_config(config_name)
        self.manager = CourseRAGManager(self.settings)
        self.manager.connect_elasticsearch()
        self.batch_size = batch_size
        self.default_k_values = [1, 3, 5, 10]

    def _prepare_queries_data(self, eval_set) -> List[Dict]:
        """Extract query, course, expected_id and filter out empty questions."""
        queries_data = []
        for idx, item in enumerate(eval_set):
            query = item['original_doc'].get('question', '')
            if not query:
                logger.warning(f"Dropping eval set item {idx} – missing 'question' field")
                continue
            queries_data.append({
                'query': query,
                'course': item['original_doc'].get('course', ''),
                'expected_id': item['expected_id']
            })
        return queries_data

    def _process_batch(self, batch: List[Dict], k: int) -> List[Dict]:
        """Perform batch search for one k and return results for that batch."""
        queries = [item['query'] for item in batch]
        courses = [item['course'] for item in batch]
        expected_ids = [item['expected_id'] for item in batch]

        # Group by course
        unique_courses = list(set(courses))
        batch_results_by_index = {}
        for course in unique_courses:
            course_indices = [i for i, c in enumerate(courses) if c == course]
            course_queries = [queries[i] for i in course_indices]
            if course_queries:
                try:
                    hits_list = self.manager.batch_search_faq(course_queries, k, course)
                    for idx_course, hits in zip(course_indices, hits_list):
                        batch_results_by_index[idx_course] = hits
                except Exception as e:
                    logger.error(f"Batch search failed for course {course}: {e}")
                    for idx_course in course_indices:
                        batch_results_by_index[idx_course] = []

        # Build results for each query in this batch
        batch_results = []
        for idx, item in enumerate(batch):
            hits = batch_results_by_index.get(idx, [])
            expected_id = item['expected_id']
            result = self._build_result_for_query(item, hits, k, expected_id)
            batch_results.append(result)
        return batch_results

    def _build_result_for_query(self, item: Dict, hits: List, k: int, expected_id: str) -> Dict:
        """Build a single result entry, including rank computation."""
        hit_ids = [hit['_id'] for hit in hits]
        # Determine rank of expected_id within top-k hits (1-indexed)
        rank = None
        for pos, hit_id in enumerate(hit_ids[:k], start=1):
            if hit_id == expected_id:
                rank = pos
                break
        success = rank is not None
        # found_id = expected_id if found, else top hit or "NONE"
        if success:
            found_id = expected_id
        else:
            found_id = hit_ids[0] if hit_ids else "NONE"

        # Build contexts (use answer field, or fallback to text)
        contexts = [hit['_source'].get('answer', hit['_source'].get('text', '')) for hit in hits]

        return {
            'k': k,
            'query': item['query'],
            'expected_id': expected_id,
            'found_id': found_id,
            'success': success,
            'rank': rank if rank else -1,   # -1 means not found
            'score': hits[0]['_score'] if hits else 0.0,
            'found_course': hits[0]['_source'].get('course', 'NONE') if hits else 'NONE',
            'contexts': contexts[:k]   # only first k contexts
        }

    def run_benchmark(self, k_values: List[int] = None) -> Dict[str, Any]:
        if k_values is None:
            k_values = self.default_k_values.copy()

        eval_set = get_eval_set_from_es()
        if not eval_set:
            raise ValueError("Eval set is empty – cannot run benchmark.")

        queries_data = self._prepare_queries_data(eval_set)
        total_queries = len(queries_data)
        total_batches = (total_queries + self.batch_size - 1) // self.batch_size
        all_results = []

        for k in k_values:
            logger.info(f"Processing k={k}")
            for batch_idx in range(0, total_queries, self.batch_size):
                batch = queries_data[batch_idx:batch_idx + self.batch_size]
                batch_num = batch_idx // self.batch_size + 1
                logger.debug(f"  Batch {batch_num}/{total_batches}")
                start_time = time.time()
                batch_results = self._process_batch(batch, k)
                latency_ms = (time.time() - start_time) * 1000 / len(batch)   # average per query
                # Attach latency to each result
                for res in batch_results:
                    res['latency_ms'] = round(latency_ms, 2)
                all_results.extend(batch_results)

        output = {
            'metadata': {
                'name': self.settings.get('name'),
                'config_name': self.config_name,
                'settings': self.settings,
                'timestamp': datetime.now().isoformat(),
                'total_queries': total_queries,
                'k_values': k_values,
                'batch_size': self.batch_size,
                'total_batches': total_batches
            },
            'results': all_results
        }
        return output

    def save_results(self, output: Dict[str, Any]) -> str:
        web_root = '/home/admin/LLM/LLM/01/web'
        results_dir = f'{web_root}/experiments/results'
        os.makedirs(results_dir, exist_ok=True)
        filename = f'{results_dir}/{self.config_name}.json'
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved results to {filename}")
        return filename

    def run_and_save(self, k_values: List[int] = None) -> str:
        return self.save_results(self.run_benchmark(k_values))


def run_benchmark(config_name: str, k_values: List[int] = None, batch_size: int = 50) -> str:
    runner = BenchmarkRunner(config_name, batch_size=batch_size)
    return runner.run_and_save(k_values)