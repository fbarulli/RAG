# /home/admin/LLM/LLM/01/web/src/experiment_pipeline.py

import sys
import os
import json
import glob
from datetime import datetime
from typing import List, Dict, Any, Optional

from langfuse.decorators import observe, langfuse_context
from src.search import CourseRAGManager
from src.config_manager import load_config
from src.run_stats import get_eval_set
from src.stats import StatsCollector

def get_experiment_name(experiment_type: str, config_name: str, variant: str = None) -> str:
    parts = [experiment_type, config_name]
    if variant:
        parts.append(variant)
    return "__".join(parts)

def get_ab_test_name(config_a: str, config_b: str) -> str:
    date_str = datetime.now().strftime("%Y-%m-%d")
    return f"ab_test__{config_a}_vs_{config_b}__{date_str}"

@observe()
def run_benchmark_experiment(config_path: str, experiment_name: str, n_per_course: int = 10):
    langfuse_context.update_current_trace(
        name=experiment_name,
        metadata={
            "config_path": config_path,
            "n_per_course": n_per_course,
            "timestamp": datetime.now().isoformat()
        },
        tags=["benchmark", "elasticsearch", "bm25"]
    )
    
    collector = StatsCollector(config_path)
    eval_set = get_eval_set("documents.json", n_per_course=n_per_course)
    result_file = collector.run_benchmark(eval_set, experiment_name)
    
    langfuse_context.update_current_trace(
        metadata={"result_file": result_file, "num_queries": len(eval_set)}
    )
    
    return result_file

@observe()
def run_ab_test_experiment(config_a: str, config_b: str, num_queries: int = 20, k: int = 3):
    experiment_name = get_ab_test_name(config_a, config_b)
    
    langfuse_context.update_current_trace(
        name=experiment_name,
        metadata={
            "config_a": config_a,
            "config_b": config_b,
            "num_queries": num_queries,
            "k": k
        },
        tags=["ab_test", "comparison"]
    )
    
    os.chdir('/home/admin/LLM/LLM/01/web')
    
    eval_set = get_eval_set("documents.json", n_per_course=num_queries // 3)
    settings_a = load_config(f"experiments/configs/{config_a}.json")
    settings_b = load_config(f"experiments/configs/{config_b}.json")
    
    manager_a = CourseRAGManager(settings_a)
    manager_b = CourseRAGManager(settings_b)
    manager_a.connect_elasticsearch()
    manager_b.connect_elasticsearch()
    
    results = []
    for idx, item in enumerate(eval_set[:num_queries]):
        res_a = manager_a.search_faq(item['query'], k, None)
        res_b = manager_b.search_faq(item['query'], k, None)
        
        result = {
            'query': item['query'],
            'expected_course': item['course'],
            'expected_id': item['expected_id'],
            'config_a_course': res_a[0]['_source']['course'] if res_a else 'NONE',
            'config_a_id': res_a[0]['_id'] if res_a else 'NONE',
            'config_a_score': round(res_a[0]['_score'], 2) if res_a else 0,
            'config_b_course': res_b[0]['_source']['course'] if res_b else 'NONE',
            'config_b_id': res_b[0]['_id'] if res_b else 'NONE',
            'config_b_score': round(res_b[0]['_score'], 2) if res_b else 0,
        }
        
        result['winner'] = 'A' if result['config_a_score'] > result['config_b_score'] else ('B' if result['config_b_score'] > result['config_a_score'] else 'TIE')
        results.append(result)
    
    a_wins = sum(1 for r in results if r['winner'] == 'A')
    b_wins = sum(1 for r in results if r['winner'] == 'B')
    ties = sum(1 for r in results if r['winner'] == 'TIE')
    
    summary = {
        'total_queries': len(results),
        'config_a_wins': a_wins,
        'config_b_wins': b_wins,
        'ties': ties,
        'winner': config_a if a_wins > b_wins else (config_b if b_wins > a_wins else 'TIE')
    }
    
    langfuse_context.update_current_trace(metadata=summary)
    
    results_dir = "experiments/ab_results"
    os.makedirs(results_dir, exist_ok=True)
    results_file = os.path.join(results_dir, f"{experiment_name}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            'metadata': {
                'experiment_name': experiment_name,
                'config_a': config_a,
                'config_b': config_b,
                'num_queries': num_queries,
                'k': k,
                'timestamp': datetime.now().isoformat()
            },
            'summary': summary,
            'results': results
        }, f, indent=4)
    
    print(f"Results saved to {results_file}")
    print(f"\n🏆 WINNER: {summary['winner']}")
    print(f"   {config_a}: {a_wins} wins")
    print(f"   {config_b}: {b_wins} wins")
    print(f"   Ties: {ties}")
    
    return results, summary

@observe()
def run_experiment_batch(config_dir: str = "experiments/configs", n_per_course: int = 10):
    experiment_name = f"batch__{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    
    langfuse_context.update_current_trace(
        name=experiment_name,
        tags=["batch", "elasticsearch_tuning"]
    )
    
    config_files = glob.glob(os.path.join(config_dir, "*.json"))
    results = {}
    
    for config_path in sorted(config_files):
        config_name = os.path.basename(config_path).replace(".json", "")
        exp_name = get_experiment_name("tuning", config_name, "benchmark")
        
        print(f"\n{'='*50}")
        print(f"Running: {config_name}")
        print(f"{'='*50}")
        
        result_file = run_benchmark_experiment(config_path, exp_name, n_per_course)
        results[config_name] = result_file
    
    langfuse_context.update_current_trace(metadata={"completed_experiments": list(results.keys())})
    
    return results

def compare_configs(config_a: str, config_b: str, num_queries: int = 20):
    """Quick A/B test between two configs."""
    return run_ab_test_experiment(config_a, config_b, num_queries)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["batch", "ab_test", "single"], default="batch")
    parser.add_argument("--config_a", default="baseline_bm25")
    parser.add_argument("--config_b", default="global_cross_fields")
    parser.add_argument("--num_queries", type=int, default=20)
    
    args = parser.parse_args()
    
    if args.mode == "batch":
        run_experiment_batch()
    elif args.mode == "ab_test":
        run_ab_test_experiment(args.config_a, args.config_b, args.num_queries)
    elif args.mode == "single":
        run_benchmark_experiment(f"experiments/configs/{args.config_a}.json", args.config_a)