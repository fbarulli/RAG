# /home/admin/LLM/LLM/01/web/notebooks/significance_test.py

import sys
import os

# Add web root to path
web_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, web_root)

from src.visualizer import RAGVisualizer
import pandas as pd
import numpy as np
from statsmodels.stats.contingency_tables import mcnemar

def run_significance_tests(k=5):
    viz = RAGVisualizer()
    registry = viz.get_experiment_registry()
    filenames = registry['filename'].tolist()
    df = viz.load_selected_experiments(filenames)
    
    # Filter to specific K
    df_k = df[df['k'] == k].copy()
    
    experiments = df_k['run_label'].unique()
    print(f"=== Statistical Significance Tests (McNemar, K={k}) ===")
    print(f"Comparing {len(experiments)} experiments\n")
    
    results = []
    
    for i, exp_a in enumerate(experiments):
        for exp_b in experiments[i+1:]:
            # Get success for each query across both experiments
            a_data = df_k[df_k['run_label'] == exp_a].set_index('query')['success']
            b_data = df_k[df_k['run_label'] == exp_b].set_index('query')['success']
            
            common_queries = a_data.index.intersection(b_data.index)
            
            if len(common_queries) == 0:
                continue
            
            # Contingency table
            a_only = sum((a_data[common_queries] == 1) & (b_data[common_queries] == 0))
            b_only = sum((a_data[common_queries] == 0) & (b_data[common_queries] == 1))
            both = sum((a_data[common_queries] == 1) & (b_data[common_queries] == 1))
            neither = sum((a_data[common_queries] == 0) & (b_data[common_queries] == 0))
            
            total = len(common_queries)
            a_recall = (both + a_only) / total
            b_recall = (both + b_only) / total
            
            if a_only + b_only > 0:
                table = [[both, a_only], [b_only, neither]]
                result = mcnemar(table, exact=False, correction=True)
                p_value = result.pvalue
                significant = p_value < 0.05
            else:
                p_value = 1.0
                significant = False
            
            results.append({
                'exp_a': exp_a,
                'exp_b': exp_b,
                'a_recall': round(a_recall, 4),
                'b_recall': round(b_recall, 4),
                'a_better': a_recall > b_recall,
                'b_better': b_recall > a_recall,
                'p_value': round(p_value, 4),
                'significant': significant
            })
    
    results_df = pd.DataFrame(results)
    
    # Show only significant differences
    sig_results = results_df[results_df['significant'] == True]
    
    if len(sig_results) > 0:
        print("\n=== SIGNIFICANT DIFFERENCES (p < 0.05) ===")
        for _, row in sig_results.iterrows():
            winner = row['exp_a'] if row['a_better'] else row['exp_b']
            loser = row['exp_b'] if row['a_better'] else row['exp_a']
            print(f"{winner} > {loser} ({row['a_recall'] if row['a_better'] else row['b_recall']} vs {row['b_recall'] if row['a_better'] else row['a_recall']}, p={row['p_value']})")
    else:
        print("\n=== No statistically significant differences found ===")
        print("All configs perform similarly at 95% confidence level")
    
    # Show best performers
    print("\n=== BEST CONFIGS BY RECALL@5 ===")
    best = df_k.groupby('run_label')['success'].mean().sort_values(ascending=False)
    print(best.round(4))
    
    return results_df

if __name__ == "__main__":
    results = run_significance_tests(k=5)