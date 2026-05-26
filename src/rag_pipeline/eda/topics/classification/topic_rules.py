"""
Hybrid Data-Driven Classification Rules.
Combines: manual seeds + TF-IDF terms + NER-derived entities.
"""
from pathlib import Path
import json
from dataclasses import dataclass
from typing import Set
from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

@dataclass
class ClassificationRules:
    error_signals: Set[str]
    admin_signals: Set[str]
    concept_signals: Set[str]

    @classmethod
    def load(cls, rules_dir: Path) -> "ClassificationRules":
        """Load seed rules + enrich from TF-IDF and NER patterns."""
        rules_path = rules_dir / "classification_rules.json"
        tfidf_path = rules_dir / "data_driven_terms.json"
        ner_path = rules_dir.parent / "entity_patterns.json"

        # Load base seeds
        if not rules_path.exists():
            logger.warning(f"Base rules not found: {rules_path}")
            error, admin, concept = set(), set(), set()
        else:
            with open(rules_path) as f:
                data = json.load(f)
            error = set(data.get("error_signals", []))
            admin = set(data.get("admin_signals", []))
            concept = set(data.get("concept_signals", []))

        # Enrich with TF-IDF
        if tfidf_path.exists():
            try:
                with open(tfidf_path) as f:
                    tf = json.load(f)
                error.update(tf.get("error_terms", [])[:20])
                admin.update(tf.get("admin_terms", [])[:20])
                concept.update(tf.get("concept_terms", [])[:20])
            except Exception as e:
                logger.warning(f"TF-IDF enrichment failed: {e}")

        # Enrich with NER patterns (best source for entities)
        if ner_path.exists():
            try:
                with open(ner_path) as f:
                    ner_data = json.load(f)
                
                # Extract high-value entities
                for pattern in ner_data.get("patterns", []):
                    label = pattern.get("label", "")
                    token = pattern.get("pattern", [{}])[0].get("LOWER", "")
                    if token:
                        if label in ["ERROR", "EXCEPTION"]:
                            error.add(token)
                        elif label in ["ADMIN", "COURSE_EVENT"]:
                            admin.add(token)
                        elif label in ["CONCEPT", "TECH_TERM"]:
                            concept.add(token)
            except Exception as e:
                logger.warning(f"NER enrichment failed: {e}")

        logger.info(f"Loaded rules → Error: {len(error)}, Admin: {len(admin)}, Concept: {len(concept)}")
        return cls(error_signals=error, admin_signals=admin, concept_signals=concept)

    def reclassify(self, current_category: str, question: str) -> str:
        """Priority: ERROR > ADMIN > CONCEPT"""
        if current_category != "OTHER":
            return current_category

        q_lower = question.lower().strip()

        if any(sig in q_lower for sig in self.error_signals):
            return "ERROR"
        if any(sig in q_lower for sig in self.admin_signals):
            return "ADMIN"
        if any(sig in q_lower for sig in self.concept_signals):
            return "CONCEPT"

        return "OTHER"
