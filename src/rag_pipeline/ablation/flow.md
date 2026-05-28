Here's the full updated plan for reference:

---

# Ablation Plan v2

## Known results (baseline)
| Experiment | H@1 | MRR | Delta vs neither |
|---|---|---|---|
| baseline | 84.9% | 0.9025 | +4.6pp |
| no_category | 88.6% | 0.9231 | +8.3pp ← current best |
| no_entity | 84.9% | 0.9025 | +4.6pp |
| neither | 80.3% | 0.8719 | — |

**Fixed**: `ner_category` removed from retriever `should` clause (line 105-106 `retrievers.py`)

---

## What to build first

### `experiment.py` — add to `Patch` dataclass
```python
GENERIC_ENTITIES = {"error", "homework", "course", "model", "project", "issue"}

@dataclass
class Patch:
    # existing
    null_entity: bool = False
    null_category: bool = False
    null_topics: bool = False
    skip_ner: bool = False
    empty_entity_patterns: bool = False
    skip_cluster: bool = False
    skip_rules: bool = False
    # new
    null_generic_entities: bool = False
    null_low_confidence_topics: bool = False
    topic_prob_threshold: float = 0.5
```

`apply_to_assignments` additions:
```python
if self.null_generic_entities:
    if a.get("ner_primary_entity") in GENERIC_ENTITIES:
        a["ner_primary_entity"] = None

if self.null_low_confidence_topics:
    if a.get("topic_probability", 1.0) < self.topic_prob_threshold:
        a["topic"] = -1
```

`label()` additions:
```python
if self.null_generic_entities:      parts.append("no_generic_entity")
if self.null_low_confidence_topics: parts.append(f"low_conf_{str(self.topic_prob_threshold).replace('.','')}")
```

### `configs/benchmark_cli.py` — add to ablation arg group
```python
g.add_argument("--null-generic-entity",          action="store_true", default=False)
g.add_argument("--null-low-confidence-topics",   action="store_true", default=False)
g.add_argument("--topic-prob-threshold",         type=float, default=0.5)
```

### `cli.py` — add to `Patch(...)` constructor
```python
null_generic_entities=args.null_generic_entity,
null_low_confidence_topics=args.null_low_confidence_topics,
topic_prob_threshold=args.topic_prob_threshold,
```

---

## Experiment queue

| # | Name | Flags | Patch path | Speed | Hypothesis |
|---|---|---|---|---|---|
| 1 | no_generic_entity | `--null-generic-entity` | payload-only | fast | generic entities hurt precision |
| 2 | no_cluster | `--skip-cluster` | rerun | slow | BERTopic cluster topics add noise |
| 3 | low_conf_null_50 | `--null-low-confidence-topics` | payload-only | fast | weak topic assignments hurt |
| 4 | low_conf_null_40 | `--null-low-confidence-topics --topic-prob-threshold 0.4` | payload-only | fast | lower threshold captures more noise |
| 5 | optimized | best of 1 + best of 3/4 | payload-only | fast | combined signal is additive |

Run in order — 2 can run overnight since it's slow.

---

## Commands
```bash
uv run python -m rag_pipeline.ablation run --name no_generic_entity --null-generic-entity
uv run python -m rag_pipeline.ablation run --name no_cluster --skip-cluster
uv run python -m rag_pipeline.ablation run --name low_conf_null_50 --null-low-confidence-topics
uv run python -m rag_pipeline.ablation run --name low_conf_null_40 --null-low-confidence-topics --topic-prob-threshold 0.4
uv run python -m rag_pipeline.ablation run --name optimized --null-generic-entity --null-low-confidence-topics  # adjust threshold based on 3 vs 4
```

---

## Report improvements (post-experiments)
- Delta vs baseline column
- Query count so small pp differences are interpretable
- `--csv` flag for export

---

## What we're NOT doing
- MLflow or external tracking — existing JSON metadata + `report` command is sufficient
- Addressing null-entity docs (400 intentionally vague/short stress-test questions)
- Re-adding `ner_category` to retriever — already confirmed harmful