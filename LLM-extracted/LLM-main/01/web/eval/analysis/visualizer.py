"""
eval/analysis/visualizer.py
============================
Visualization for test_variations results.
Saves plots to experiments/plots/ directory.

Run:    uv run python eval/analysis/visualizer.py
"""
import sys, os, json, glob
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving files

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = 'experiments/results'
PLOTS_DIR = 'experiments/plots'


def load_data():
    files = sorted(glob.glob(f'{RESULTS_DIR}/variations_*.json'))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f)


def plot_recall(data, save=True):
    configs = data['configs']
    names = [c['name'] for c in configs]
    r1 = [c['recall_at_k']['1'] * 100 for c in configs]
    r5 = [c['recall_at_k']['5'] * 100 for c in configs]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, r1, width, label='Recall@1', color='steelblue')
    ax.bar(x + width/2, r5, width, label='Recall@5', color='orange')
    
    for bar in ax.patches:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.5, f'{h:.1f}%',
                    ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Recall (%)')
    ax.set_title('Retrieval Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/recall_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR}/recall_comparison.png")
    plt.close()


def plot_latency(data, save=True):
    configs = data['configs']
    names = [c['name'] for c in configs]
    p50 = [c['p50_latency'] for c in configs]
    p95 = [c['p95_latency'] for c in configs]
    
    x = np.arange(len(names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, p50, width, label='P50', color='green')
    ax.bar(x + width/2, p95, width, label='P95', color='red')
    
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Latency Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/latency_comparison.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR}/latency_comparison.png")
    plt.close()


def plot_strategy(data, save=True):
    best = max(data['configs'], key=lambda c: c['recall_at_k']['5'])
    if not best.get('per_strategy'):
        return
    
    strategies = list(best['per_strategy'].keys())
    r5 = [best['per_strategy'][s]['recall_at_k']['5'] * 100 for s in strategies]
    colors = ['#2ecc71', '#3498db', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(strategies, r5, color=colors[:len(strategies)])
    
    for bar, val in zip(bars, r5):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.5, f'{val:.1f}%', ha='center', fontsize=11)
    
    ax.set_ylabel('Recall@5 (%)')
    ax.set_title(f'Per-Strategy ({best["name"]})')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/strategy_breakdown.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR}/strategy_breakdown.png")
    plt.close()


def plot_course(data, save=True):
    best = max(data['configs'], key=lambda c: c['recall_at_k']['5'])
    if not best.get('per_course'):
        return
    
    courses = list(best['per_course'].keys())
    r5 = [best['per_course'][c]['recall_at_k']['5'] * 100 for c in courses]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(courses, r5, color=sns.color_palette('Set2', len(courses)))
    
    for bar, val in zip(bars, r5):
        ax.text(bar.get_x() + bar.get_width()/2., val + 0.5, f'{val:.1f}%', ha='center', fontsize=11)
    
    ax.set_ylabel('Recall@5 (%)')
    ax.set_title(f'Per-Course ({best["name"]})')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save:
        os.makedirs(PLOTS_DIR, exist_ok=True)
        plt.savefig(f'{PLOTS_DIR}/course_breakdown.png', dpi=150, bbox_inches='tight')
        print(f"Saved: {PLOTS_DIR}/course_breakdown.png")
    plt.close()


def plot_all():
    data = load_data()
    if not data:
        print("No variations data. Run: uv run python eval/benchmarks/test_variations.py")
        return
    
    print(f"Generating plots from {len(data['configs'])} configs...")
    plot_recall(data)
    plot_latency(data)
    plot_strategy(data)
    plot_course(data)
    print(f"\nAll plots saved to {PLOTS_DIR}/")


if __name__ == '__main__':
    plot_all()
