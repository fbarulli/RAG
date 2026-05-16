"""
_topic_ner_diagnostics.py
=========================
Diagnostic module for analyzing OTHER questions and suggesting pattern fixes.

Usage
-----
    cd /workspaces/LLM && uv run python -m production_pipeline.p02_eda._topic_ner_diagnostics
"""
import json
import re
from collections import Counter
from pathlib import Path

import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from production_pipeline.p02_eda._entity_pattern_learner import (
    build_base_nlp,
    extract_missed_terms,
    suggest_patterns,
    update_entity_ruler,
)

DATA_PATH = Path("production_pipeline/p02_eda/experiments/topic_assignments_BAAI_bge_base_en_v1.5.json")

# Error signals for reclassification
_ERROR_SIGNALS = {
    "error", "exception", "failed", "failure", "warning", "cannot", "can't",
    "unable", "not found", "permission denied", "attributeerror", "valueerror",
    "typeerror", "importerror", "modulenotfounderror", "filenotfounderror",
    "keyerror", "oserror", "runtimeerror", "nameerror", "traceback",
    "convergencewarning", "futurewarning", "userwarning", "deprecationwarning",
    "timeout", "refused", "denied", "crash", "invalid", "unrecognized",
    "could not", "no module", "no such",
}

_ADMIN_SIGNALS = {
    "certificate", "homework", "deadline", "cohort", "office hours",
    "self-paced", "graduate", "leaderboard", "peer review", "capstone",
    "lecture", "video", "live", "recorded", "session", "weeks", "module",
    "form", "confirmation", "email", "registration", "enroll",
}

_CONCEPT_SIGNALS = {
    "what is", "why do", "why does", "what does", "difference between",
    "how does", "explain", "understanding", "when to use", "what are",
    "how to choose", "what's the", "why use", "should i use",
}


def reclassify_others(assignments: list[dict], nlp) -> dict[str, int]:
    """
    Attempt rule-based reclassification of OTHER questions.
    Returns category -> count after reclassification.
    """
    reclassified = Counter()
    still_other = []

    for a in assignments:
        if a.get("ner_category") != "OTHER":
            continue

        q = a["question"].lower()

        if any(sig in q for sig in _ERROR_SIGNALS):
            reclassified["ERROR"] += 1
        elif any(sig in q for sig in _ADMIN_SIGNALS):
            reclassified["ADMIN"] += 1
        elif any(sig in q for sig in _CONCEPT_SIGNALS):
            reclassified["CONCEPT"] += 1
        else:
            still_other.append(a)

    return reclassified, still_other


def analyze_still_other(still_other: list[dict], nlp) -> None:
    """Extract top unmatched terms from remaining OTHER questions."""
    questions = [a["question"] for a in still_other]
    missed = Counter()
    for doc in nlp.pipe(questions, batch_size=64):
        words = re.findall(r'\b[a-zA-Z][a-zA-Z]{2,}\b', doc.text)
        filtered = [
            w.lower() for w in words
            if w.lower() not in STOP_WORDS
            and len(w) > 2
            and not w.isupper()  # skip acronyms
        ]
        missed.update(filtered)

    print(f"\n{'='*60}")
    print(f"TOP UNMATCHED TERMS IN REMAINING OTHER ({len(still_other)} questions)")
    print(f"{'='*60}")
    for term, count in missed.most_common(40):
        print(f"  {count:3d}  {term}")


def main():
    data = json.load(open(DATA_PATH))
    assignments = data["assignments"]
    others = [a for a in assignments if a.get("ner_category") == "OTHER"]

    print(f"\n{'='*60}")
    print(f"OTHER ANALYSIS: {len(others)}/{len(assignments)} questions")
    print(f"{'='*60}")

    nlp = build_base_nlp()
    missed = extract_missed_terms([a["question"] for a in assignments], nlp)
    suggestions = suggest_patterns(missed, min_count=3)
    nlp = update_entity_ruler(nlp, suggestions)

    # Reclassify via signals
    reclassified, still_other = reclassify_others(others, nlp)

    print(f"\nRule-based reclassification of {len(others)} OTHER questions:")
    for cat, count in reclassified.most_common():
        print(f"  -> {cat:10s} {count:3d}")
    print(f"  -> STILL OTHER {len(still_other):3d}")

    # Sample still-other
    print(f"\nSample of {min(20, len(still_other))} remaining OTHER questions:")
    for a in still_other[:20]:
        print(f"  [topic={a['topic']:3d}] {a['question']}")

    # Analyze unmatched terms
    analyze_still_other(still_other, nlp)

    # Summary suggestion
    print(f"\n{'='*60}")
    print("SUGGESTED ADDITIONS TO base_patterns")
    print(f"{'='*60}")
    print("Review the top unmatched terms above and add to:")
    print("  TOOL    -> infrastructure/framework names")
    print("  CONCEPT -> ML/data concepts")
    print("  ERROR   -> add signals to _ERROR_SIGNALS in this file")
    print("  ADMIN   -> add signals to _ADMIN_SIGNALS in this file")


if __name__ == "__main__":
    main()