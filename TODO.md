# Ablation experiments — progress log

## Status: in progress

---

## Known results (BAAI/bge-base-en-v1.5, entity_boosted config)

| Experiment | H@1 | MRR | Delta vs neither | Notes |
|---|---|---|---|---|
| neither | 80.3% | 0.8719 | — | pure vector |
| baseline | 84.9% | 0.9025 | +4.6pp | |
| no_entity | 84.9% | 0.9025 | +4.6pp | entity adds nothing alone |
| no_category | 88.6% | 0.9231 | +8.3pp | ← was optimal |
| no_generic_entity | 85.5% | 0.9054 | +5.2pp | +0.6pp vs baseline |
| low_conf_null_50 | 88.6% | 0.9231 | +8.3pp | no effect — topic not in retriever |
| low_conf_null_40 | pending | | | |
| no_cluster | pending | | | slow rerun |
| optimized | pending | | | best combination |

---

## Key findings so far

- `ner_category` in Qdrant `should` clause hurts by 3.7pp — already removed from `retrievers.py` line 105-106
- Generic entity nulling (`error`, `homework`, `course`, `model`, `project`, `issue`) gives +0.6pp — mild noise, worth keeping
- Low-confidence topic nulling (threshold 0.5) has zero effect — `topic` field is not used in the retriever should clause, so payload changes to it are irrelevant at query time
- 400 null-entity docs are intentional stress-test questions — not addressable

---

## What was built

### New patch flags (payload-only, no rerun)
- `--null-generic-entity` — nulls `ner_primary_entity` for values in `GENERIC_ENTITIES` blocklist
- `--null-low-confidence-topics` — sets `topic=-1` for docs where `topic_probability < threshold`
- `--topic-prob-threshold` — float, default 0.5

### Files changed
- `src/rag_pipeline/ablation/experiment.py` — `GENERIC_ENTITIES` constant, two new `Patch` fields, `apply_to_assignments`, `label()`
- `configs/benchmark_cli.py` — three new CLI flags
- `src/rag_pipeline/ablation/cli.py` — wired into `Patch` constructor, added per-query-type breakdown printed after every run

---

## Experiment queue

| # | Name | Flags | Path | Status |
|---|---|---|---|---|
| 1 | no_generic_entity | `--null-generic-entity` | payload-only | ✅ done |
| 2 | low_conf_null_50 | `--null-low-confidence-topics` | payload-only | ✅ done — no effect |
| 3 | low_conf_null_40 | `--null-low-confidence-topics --topic-prob-threshold 0.4` | payload-only | 🔄 running |
| 4 | no_cluster | `--skip-cluster` | rerun | ⏳ pending (slow) |
| 5 | optimized | `--null-generic-entity` + best of 3/4 | payload-only | ⏳ pending |

---

## Outstanding questions
- Does query-type breakdown show in run output? (JSONL path match unconfirmed)
- Will low_conf_null_40 differ from 50? Unlikely given topic is not in retriever
- Does no_cluster change entity/category assignments enough to shift H@1?