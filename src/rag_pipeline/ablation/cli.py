# rag_pipeline/ablation/cli.py
"""
Ablation CLI.

Commands:
    run     -- run a named experiment
    compare -- diff two experiments query-by-query
    report  -- print summary table of all experiments

Examples:
    uv run python -m rag_pipeline.ablation run --name baseline
    uv run python -m rag_pipeline.ablation run --name no_category --null-category
    uv run python -m rag_pipeline.ablation run --name no_entity --null-entity
    uv run python -m rag_pipeline.ablation run --name neither --null-category --null-entity
    uv run python -m rag_pipeline.ablation compare baseline__entity_boosted no_category__entity_boosted
    uv run python -m rag_pipeline.ablation compare baseline__entity_boosted no_category__entity_boosted --show losses
    uv run python -m rag_pipeline.ablation report
"""
from configs.benchmark_cli import create_ablation_parser


def cmd_run(args) -> None:
    from rag_pipeline.ablation.experiment import Experiment, Patch
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
            null_generic_entities=args.null_generic_entity,
            null_low_confidence_topics=args.null_low_confidence_topics,
            topic_prob_threshold=args.topic_prob_threshold,
        ),
        configs=args.configs,
        model=args.model,
    )
    result = exp.run()
    print(f"\nExperiment '{result.name}' complete")
    for cfg, metrics in result.metrics.items():
        h1  = metrics.get("h1",  metrics.get("hit_at_1", "?"))
        mrr = metrics.get("mrr", "?")
        h1_str  = f"{h1:.1%}"  if isinstance(h1,  float) else str(h1)
        mrr_str = f"{mrr:.4f}" if isinstance(mrr, float) else str(mrr)
        print(f"  {cfg:<25} H@1={h1_str}  MRR={mrr_str}")

    # per-query-type breakdown from saved JSONL
    import json
    from collections import defaultdict
    from rag_pipeline.core.paths import Paths
    results_dir = Paths.ablation_results_dir()
    for cfg in result.configs:
        jsonl = results_dir / f"{result.name}__{cfg}_query_results.jsonl"
        if not jsonl.exists():
            continue
        buckets = defaultdict(list)
        with open(jsonl) as f:
            for line in f:
                row = json.loads(line)
                qt = row.get("query_type", "unknown")
                hit = row.get("hit_ids", [])
                expected = row.get("expected_id", "")
                buckets[qt].append(int(bool(hit and hit[0] == expected)))
        print(f"\n  [{cfg}] breakdown by query type:")
        print(f"  {'query_type':<22} {'n':>5} {'H@1':>7}")
        print(f"  {'-' * 36}")
        for qt in sorted(buckets):
            hits = buckets[qt]
            print(f"  {qt:<22} {len(hits):>5} {sum(hits)/len(hits):>7.1%}")


def cmd_compare(args) -> None:
    from rag_pipeline.ablation.compare import compare, summary, print_diff
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
    from rag_pipeline.ablation.report import print_report
    print_report(ci=getattr(args, "ci", False))


def main() -> None:
    parser = create_ablation_parser()
    args = parser.parse_args()

    dispatch = {
        "run":     cmd_run,
        "compare": cmd_compare,
        "report":  cmd_report,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()