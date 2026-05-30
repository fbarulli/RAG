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
        h1  = metrics.hit_rate_1
        mrr = metrics.mrr
        h1_str  = f"{h1:.1%}"  if isinstance(h1,  float) else str(h1)
        mrr_str = f"{mrr:.4f}" if isinstance(mrr, float) else str(mrr)
        print(f"  {cfg:<25} H@1={h1_str}  MRR={mrr_str}")

    # per-query-type breakdown from saved JSONL
    from rag_pipeline.ablation.compare import breakdown_from_jsonl
    from rag_pipeline.core.paths import Paths
    results_dir = Paths.ablation_results_dir()
    for cfg in result.configs:
        jsonl = results_dir / f"{result.name}__{cfg}_query_results.jsonl"
        print(f"\n  [{cfg}] breakdown by query type:")
        breakdown_from_jsonl(jsonl)


def cmd_flow(args) -> None:
    from rag_pipeline.ablation.experiment import Experiment, Patch
    from rag_pipeline.core.paths import Paths

    defaults = {}
    if args.configs:
        defaults["configs"] = args.configs
    if args.model:
        defaults["model"] = args.model
    if getattr(args, "encode_mode", None):
        defaults["encode_mode"] = args.encode_mode

    # fast payload-only experiments first
    fast = [
        ("baseline",           Patch()),
        ("no_entity",          Patch(null_entity=True)),
        ("no_category",        Patch(null_category=True)),
        ("no_topics",          Patch(null_topics=True)),
        ("no_generic_entity",  Patch(null_generic_entities=True)),
        ("low_conf_null_50",   Patch(null_low_confidence_topics=True, topic_prob_threshold=0.5)),
        ("low_conf_null_40",   Patch(null_low_confidence_topics=True, topic_prob_threshold=0.4)),
    ]

    # slow rerun experiments — only if --rerun passed
    slow = [
        ("no_cluster",       Patch(skip_cluster=True)),
        ("no_rules",         Patch(skip_rules=True)),
        ("empty_patterns",   Patch(empty_entity_patterns=True)),
    ]

    suite = fast + (slow if args.rerun else [])
    print(f"Running {len(suite)} experiments ({len(fast)} fast" + (f", {len(slow)} slow" if args.rerun else ", rerun skipped") + ")")

    for name, patch in suite:
        print("\n" + "="*50)
        exp = Experiment(name=name, patch=patch, **defaults)
        try:
            result = exp.run()
            for cfg, metrics in result.metrics.items():
                h1  = metrics.hit_rate_1
                mrr = metrics.mrr
                h1_str  = f"{h1:.1%}"  if isinstance(h1,  float) else str(h1)
                mrr_str = f"{mrr:.4f}" if isinstance(mrr, float) else str(mrr)
                print(f"  {name:<25} {cfg:<25} H@1={h1_str}  MRR={mrr_str}")
        except Exception as e:
            print(f"  ERROR in {name}: {e}")
            continue
        from rag_pipeline.ablation.compare import breakdown_from_jsonl
        from rag_pipeline.core.paths import Paths
        results_dir = Paths.ablation_results_dir()
        for cfg in result.configs:
            jsonl = results_dir / f"{result.name}__{cfg}_query_results.jsonl"
            print(f"\n  [{cfg}] breakdown by query type:")
            breakdown_from_jsonl(jsonl)

    print("\n" + "="*50)
    print("Flow complete. Run: uv run python -m rag_pipeline.ablation report")


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
        "flow":    cmd_flow,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()