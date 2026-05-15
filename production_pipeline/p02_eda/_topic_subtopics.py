"""
_topic_subtopics.py
===================
Generate subtopics for large parent topics using BERTopic.

Single responsibility: recursive clustering on topic subsets.
No document loading, no main model fitting, no output serialization.

Functions:
    build_subtopics(assignments, questions, embeddings, subtopic_threshold, subtopic_min_size) -> dict
"""
from collections import Counter
from typing import TypedDict

import numpy as np
from bertopic import BERTopic

from rag_pipeline.logging import get_logger

logger = get_logger(__name__)

# Passed to BERTopic to signal that embeddings are pre-computed and no internal
# embedding model should be initialised.  Using the string sentinel avoids the
# deprecation warning / TypeError that arises from passing None in recent
# BERTopic releases.
_PRECOMPUTED = "precomputed"

# Number of keywords to retain per subtopic.
_KEYWORDS_PER_SUBTOPIC = 5


class SubtopicRecord(TypedDict):
    subtopic: int
    subtopic_keywords: list[str]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def build_subtopics(
    assignments: list,
    questions: list[str],
    embeddings: np.ndarray,
    subtopic_threshold: int,
    subtopic_min_size: int,
) -> dict[str, SubtopicRecord]:
    """
    Generate subtopics for parent topics that exceed size threshold.

    Args:
        assignments:         List of TopicAssignment dataclasses.
        questions:           List of question strings aligned with ``assignments``.
        embeddings:          Pre-computed embeddings array aligned with ``questions``.
                             Must be a 2-D NumPy array; contiguous layout is
                             required for correct fancy-index slicing.
        subtopic_threshold:  Generate subtopics only for topics larger than this.
        subtopic_min_size:   Minimum docs per subtopic in recursive clustering.

    Returns:
        Dict mapping ``str(doc_id)`` → ``{subtopic, subtopic_keywords}``.
        String keys are used throughout so the result round-trips through JSON
        without silent key-type coercion.
    """
    if not isinstance(embeddings, np.ndarray) or embeddings.ndim != 2:
        raise ValueError(
            f"embeddings must be a 2-D np.ndarray, got {type(embeddings).__name__}"
            + (f" with ndim={embeddings.ndim}" if isinstance(embeddings, np.ndarray) else "")
        )

    topic_sizes: Counter[int] = Counter(a.topic for a in assignments)

    large_topics = {
        t for t, size in topic_sizes.items()
        if t != -1 and size > subtopic_threshold
    }

    if not large_topics:
        logger.info("No topics exceed subtopic threshold; skipping subtopic generation")
        return {}

    logger.info(f"Generating subtopics for {len(large_topics)} large topics")

    outlier_count = topic_sizes.get(-1, 0)
    if outlier_count:
        logger.info(f"{outlier_count} outlier docs (topic=-1) skipped for subtopic generation")

    subtopic_cache: dict[str, SubtopicRecord] = {}

    for i, parent_topic in enumerate(sorted(large_topics), 1):
        logger.info(f"Subtopic generation [{i}/{len(large_topics)}] topic {parent_topic}")

        parent_indices = [idx for idx, a in enumerate(assignments) if a.topic == parent_topic]
        if not parent_indices:
            continue

        parent_questions = [questions[idx] for idx in parent_indices]
        # np.ascontiguousarray ensures fancy-index slices are always safe to
        # pass back into C-extension code inside BERTopic / UMAP.
        parent_embeddings = np.ascontiguousarray(embeddings[parent_indices])

        sub_model = BERTopic(
            embedding_model=_PRECOMPUTED,
            min_topic_size=subtopic_min_size,
            verbose=False,
        )

        try:
            sub_topics, _ = sub_model.fit_transform(parent_questions, parent_embeddings)
        except Exception as exc:
            logger.warning(
                f"Subtopic clustering failed for topic {parent_topic} "
                f"({len(parent_indices)} docs): {exc} — skipping"
            )
            continue

        for local_idx, global_idx in enumerate(parent_indices):
            doc_id = str(assignments[global_idx].id)
            sub_t = int(sub_topics[local_idx])

            # BERTopic returns False (not an empty list) for unknown / outlier
            # topic IDs such as -1.  Guard before slicing.
            raw_topic = sub_model.get_topic(sub_t)
            keywords: list[str] = (
                [word for word, _ in raw_topic[:_KEYWORDS_PER_SUBTOPIC]]
                if raw_topic is not False
                else []
            )

            subtopic_cache[doc_id] = SubtopicRecord(
                subtopic=sub_t,
                subtopic_keywords=keywords,
            )

    logger.info(
        f"Subtopic generation complete: {len(subtopic_cache)} docs assigned subtopics"
    )
    return subtopic_cache