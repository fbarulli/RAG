# ablation/cli.py
"""
Ablation CLI.

Commands:
    run     -- run a named experiment
    compare -- diff two experiments query-by-query
    report  -- print summary table of all experiments

Examples:
    uv run python -m ablation run --name baseline
    uv run python -m ablation run --name no_category --null-category
    uv run python -m ablation run --name no_entity --null-entity
    uv run python -m ablation run --name neither --null-category --null-entity
    uv run python -m ablation compare baseline__entity_boosted no_category__entity_boosted
    uv run python -m ablation compare baseline__entity_boosted no_category__entity_boosted --show losses
    uv run python -m ablation report
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def cmd_run(args) -> None:
    from ablation.experiment import Experiment, Patch
    exp = Experiment(
        name=args.name,
        patch=Patch(
            null_entity=args.null_entity,
            null_category=args.null_category,
            null_topics=args.null_topics,
            skip_ner=args.skip_ner,
            empty_entity_patterns=args.empty_entity_patterns,
            skip_cluster=args.skip_cluster,
            skip_rules=args.skip_rules,
        ),
        configs=args.configs,
        model=args.model,
    )
    result = exp.run()
    print(f"\nExperiment '{result.name}' complete")
    for cfg, metrics in result.metrics.items():
        h1  = metrics.get("h1",  metrics.get("hit_at_1",  "?"))
        mrr = metrics.get("mrr", "?")
        h1_str  = f"{h1:.1%}"  if isinstance(h1,  float) else str(h1)
        mrr_str = f"{mrr:.4f}" if isinstance(mrr, float) else str(mrr)
        print(f"  {cfg:<25} H@1={h1_str}  MRR={mrr_str}")


def cmd_compare(args) -> None:
    from ablation.compare import compare, summary, print_diff
    rows = compare(args.exp_a, args.exp_b, top_k=args.top_k)
    s = summary(rows)
    print(f"\n{'Metric':<20} {'Exp A':>8} {'Exp B':>8} {'Delta':>8}")
    print("-" * 48)
    print(f"{'H@1':<20} {s['h1_a']:>8.1%} {s['h1_b']:>8.1%} {s['delta_h1']:>+8.1%}")
    print(f"{'b beats a':<20} {s['b_beats_a']:>8}")
    print(f"{'a beats b':<20} {s['a_beats_b']:>8}")
    print(f"{'neutral':<20} {s['neutral']:>8}")
    print()
    print_diff(rows, show=args.show)


def cmd_report(args) -> None:
    from ablation.report import print_report
    print_report()


def main() -> None:
    parser = argparse.ArgumentParser(prog="ablation")
    sub = parser.add_subparsers(dest="command", required=True)

    # run
    run_p = sub.add_parser("run", help="Run a named experiment")
    run_p.add_argument("--name", required=True, help="Experiment name (used for output files)")
    run_p.add_argument("--null-entity",           action="store_true", help="Set ner_primary_entity=None")
    run_p.add_argument("--null-category",         action="store_true", help="Set ner_category=OTHER")
    run_p.add_argument("--null-topics",           action="store_true", help="Set topic_id=-1 in payload")
    run_p.add_argument("--skip-ner",              action="store_true", help="Null entity+category (no re-run)")
    run_p.add_argument("--empty-entity-patterns", action="store_true", help="Wipe entity_patterns.json (re-run)")
    run_p.add_argument("--skip-cluster",          action="store_true", help="Disable cluster majority (re-run)")
    run_p.add_argument("--skip-rules",            action="store_true", help="Disable keyword rules (re-run)")
    run_p.add_argument("--configs", nargs="+", default=["entity_boosted", "vector_default"])
    run_p.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    run_p.set_defaults(func=cmd_run)

    # compare
    cmp_p = sub.add_parser("compare", help="Diff two experiment result files")
    cmp_p.add_argument("exp_a", help="e.g. baseline__entity_boosted")
    cmp_p.add_argument("exp_b", help="e.g. no_category__entity_boosted")
    cmp_p.add_argument("--top-k", type=int, default=1)
    cmp_p.add_argument("--show", choices=["all", "wins", "losses"], default="all")
    cmp_p.set_defaults(func=cmd_compare)

    # report
    rep_p = sub.add_parser("report", help="Print summary table")
    rep_p.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()