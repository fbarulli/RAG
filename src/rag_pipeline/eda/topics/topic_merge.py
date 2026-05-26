"""
Merge per-model topic assignments and apply final classification rules.
"""
import json
from pathlib import Path
from collections import Counter
from rag_pipeline.logging import get_logger
from .topic_rules import ClassificationRules

logger = get_logger(__name__)

EXP_DIR = Path(__file__).parent / "experiments"
OUTPUT_PATH = EXP_DIR / "topic_assignments_all.json"


def merge(exp_dir: Path = EXP_DIR, output_path: Path = OUTPUT_PATH) -> None:
    rules = ClassificationRules.load(exp_dir.parent / "rules")

    merged = {'models': [], 'results': {}}

    files = sorted(
        f for f in exp_dir.glob("topic_assignments_*.json")
        if f.name != "topic_assignments_all.json" and "_validated" not in f.name
    )

    for f in files:
        with open(f) as fh:
            data = json.load(fh)

        model = data.get('metadata', {}).get('model', f.stem)
        reclassified = 0

        for a in data.get('assignments', []):
            original = a.get('ner_category', 'OTHER')
            updated = rules.reclassify(original, a.get('question', ''))
            if updated != original:
                a['ner_category'] = updated
                reclassified += 1

        logger.info(f"[merge] {model}: {len(data.get('assignments', []))} assignments, {reclassified} reclassified")

        merged['models'].append(model)
        merged['results'][model] = data   # Keep full original data per model

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as fh:
        json.dump(merged, fh, indent=2)

    logger.info(f"Saved merged file with {len(merged['models'])} models")

    # Summary
    print(f"\n{'='*70}")
    print(f"Merged {len(merged['models'])} models → {output_path.name}")
    print(f"{'='*70}")

    for model in merged['models']:
        assignments = merged['results'][model].get('assignments', [])
        n = len(assignments)
        if n == 0: continue
        cats = Counter(a.get('ner_category', 'MISSING') for a in assignments)
        outliers = sum(1 for a in assignments if a.get('topic') == -1)

        print(f"\n {model}  ({n} questions)")
        print(f"   outliers : {outliers} ({outliers / n:.1%})")
        for cat, count in cats.most_common():
            print(f"   {cat:12s} {count:5d} ({count / n:.1%})")


if __name__ == "__main__":
    merge()
