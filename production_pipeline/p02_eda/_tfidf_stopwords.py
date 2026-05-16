"""
/workspaces/LLM/production_pipeline/p02_eda/_tfidf_stopwords.py
========================================
Corpus-derived stopword list building — no hardcoded domain lists.

Three data-driven passes (each a superset of the previous):
  Pass 1 — chunk-level frequency       (df required)
  Pass 2 — cross-course saturation     (tfidf_results required)
  Pass 3 — IDF floor                   (fitted vectorizer required)
  Pass 4 — EDA-confirmed noise         (eda_stats dict required)

Typical call pattern
--------------------
    from rag_pipeline.preprocessing.stopwords import build_stopword_list

    # before first TF-IDF fit
    stops_v1 = build_stopword_list(df=df, eda_stats=eda_stats)

    # after first fit
    stops_v2 = build_stopword_list(
        df=df, eda_stats=eda_stats,
        tfidf_results=results_v1, vectorizer=vec_v1,
    )
"""
import logging
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer,
    ENGLISH_STOP_WORDS,
)

logger = logging.getLogger(__name__)

# Terms that are unambiguously tool/framework names and must NEVER become
# stopwords regardless of corpus frequency.  These are the bridge concepts
# we want to characterise, not suppress.  Extend as the corpus grows.
BRIDGE_CONCEPT_WHITELIST: frozenset[str] = frozenset({
    # confirmed from top_words EDA
    "docker", "python", "gcp", "dbt", "aws", "bigquery", "mlflow",
    "spark", "kestra", "pandas", "jupyter", "api", "github", "pipenv",
    # likely cross-course frameworks
    "airflow", "kafka", "kubernetes", "terraform", "postgres", "sql",
    "flask", "fastapi", "prefect", "mage", "dagster",
    "langchain", "openai", "huggingface", "bert", "llm", "rag",
    "sklearn", "xgboost", "lightgbm",
    # operational retrieval terms — must survive IDF floor
    "deploy", "deployment", "monitor", "monitoring", "production",
    "run", "pipeline", "train", "training", "inference", "serve",
    "debug", "error", "install", "configure", "setup",
})


# ─────────────────────────────────────────────────────────────────────────────
# PASS 1 — chunk-level frequency
# ─────────────────────────────────────────────────────────────────────────────
def frequency_based_stopwords(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    presence_threshold: float = 0.85,
    min_chunk_freq: int = 5,
) -> set[str]:
    """
    Terms present in >= presence_threshold fraction of chunks are corpus-wide
    noise.  More granular than max_df because results are inspectable and
    whitelisted terms are explicitly protected.

    Parameters
    ----------
    df                  : DataFrame containing the text corpus
    text_col            : column name holding cleaned text
    presence_threshold  : fraction of chunks a term must appear in [0, 1]
    min_chunk_freq      : minimum raw chunk count (filters hapax legomena)

    Returns
    -------
    set[str] of stopword strings, excluding BRIDGE_CONCEPT_WHITELIST
    """
    try:
        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )
        if df.empty:
            raise ValueError("DataFrame is empty — cannot derive frequency stopwords.")
        if not (0.0 < presence_threshold <= 1.0):
            raise ValueError(
                f"presence_threshold must be in (0, 1], got {presence_threshold}"
            )

        vec = CountVectorizer(
            ngram_range=(1, 1),
            min_df=min_chunk_freq,
            stop_words="english",
        )
        matrix        = vec.fit_transform(df[text_col])
        feature_names = vec.get_feature_names_out()

        doc_freq = np.asarray((matrix > 0).sum(axis=0)).flatten()
        presence = doc_freq / len(df)

        stopwords = {
            feature_names[i]
            for i, p in enumerate(presence)
            if p >= presence_threshold
            and feature_names[i] not in BRIDGE_CONCEPT_WHITELIST
        }

        
        logger.info(
            f"[frequency_based_stopwords] "
            f"threshold={presence_threshold}, min_df={min_chunk_freq} "
            f"-> {len(stopwords)} stopwords from {len(df)} documents"
        )
        
        logger.debug(f"Sample: {sorted(stopwords)[:20]}")
        return stopwords

    except Exception:
        logger.error(
            "[frequency_based_stopwords] Failed.\n" + traceback.format_exc()
        )
        raise


# ─────────────────────────────────────────────────────────────────────────────
# PASS 2 — cross-course saturation
# ─────────────────────────────────────────────────────────────────────────────
def saturation_based_stopwords(
    tfidf_results: dict[str, list[tuple[str, float]]],
    top_n: int = 50,
    presence_threshold: float = 0.6,
) -> set[str]:
    """
    Terms appearing in the top-N of >= presence_threshold fraction of courses
    are cross-course noise.  Run as a second pass after an initial TF-IDF fit.

    Note: legitimate bridge concepts will also be caught if they exceed the
    threshold — they are protected by BRIDGE_CONCEPT_WHITELIST.

    Parameters
    ----------
    tfidf_results       : {course: [(term, score), ...]} from fit_tfidf()
    top_n               : how many top terms per course to consider
    presence_threshold  : fraction of courses [0, 1]

    Returns
    -------
    set[str] of stopword strings, excluding BRIDGE_CONCEPT_WHITELIST
    """
    try:
        if not tfidf_results:
            raise ValueError("tfidf_results is empty.")

        n_courses = len(tfidf_results)
        if n_courses < 2:
            logger.warning(
                "[saturation_based_stopwords] Only one course found — "
                "returning empty set."
            )
            return set()

        term_count: dict[str, int] = defaultdict(int)
        for course, terms in tfidf_results.items():
            if not terms:
                logger.warning(
                    f"[saturation_based_stopwords] Course '{course}' "
                    f"has an empty term list — skipping."
                )
                continue
            for term, _ in terms[:top_n]:
                term_count[term] += 1

        cutoff    = max(2, int(n_courses * presence_threshold))
        stopwords = {
            term
            for term, count in term_count.items()
            if count >= cutoff
            and term not in BRIDGE_CONCEPT_WHITELIST
        }

        logger.info(
            f"[saturation_based_stopwords] "
            f"top_n={top_n}, threshold={presence_threshold} "
            f"(cutoff={cutoff}/{n_courses} courses) "
            f"-> {len(stopwords)} stopwords"
        )
        logger.debug(f"Sample: {sorted(stopwords)[:20]}")
        return stopwords

    except Exception:
        logger.error(
            "[saturation_based_stopwords] Failed.\n" + traceback.format_exc()
        )
        raise


# ─────────────────────────────────────────────────────────────────────────────
# PASS 3 — IDF floor
# ─────────────────────────────────────────────────────────────────────────────
def idf_floor_stopwords(
    vectorizer: TfidfVectorizer,
    percentile: float = 3.0,
) -> set[str]:
    """
    Terms with the lowest IDF scores are the most universal — extract directly
    from the fitted vectorizer rather than recomputing from scratch.

    Parameters
    ----------
    vectorizer  : fitted TfidfVectorizer (must have idf_ attribute)
    percentile  : bottom N-th percentile of IDF scores -> stopwords

    Returns
    -------
    set[str] of stopword strings, excluding BRIDGE_CONCEPT_WHITELIST
    """
    try:
        if not hasattr(vectorizer, "idf_"):
            raise ValueError(
                "Vectorizer has not been fitted yet — call fit_transform first."
            )
        if not (0.0 < percentile < 100.0):
            raise ValueError(
                f"percentile must be in (0, 100), got {percentile}"
            )

        feature_names = vectorizer.get_feature_names_out()
        idf_scores    = vectorizer.idf_
        threshold     = float(np.percentile(idf_scores, percentile))

        stopwords = {
            feature_names[i]
            for i, score in enumerate(idf_scores)
            if score <= threshold
            and feature_names[i] not in BRIDGE_CONCEPT_WHITELIST
        }

        logger.info(
            f"[idf_floor_stopwords] "
            f"percentile={percentile}, idf_threshold={threshold:.4f} "
            f"-> {len(stopwords)} stopwords"
        )
        logger.debug(f"Sample: {sorted(stopwords)[:20]}")
        return stopwords

    except Exception:
        logger.error(
            "[idf_floor_stopwords] Failed.\n" + traceback.format_exc()
        )
        raise


# ─────────────────────────────────────────────────────────────────────────────
# PASS 4 — EDA-confirmed noise
# ─────────────────────────────────────────────────────────────────────────────
def eda_derived_stopwords(
    eda_stats: dict,
    top_words_keep_pct: float = 0.4,
) -> set[str]:
    """
    Derive stopwords from the EDA output JSON (top_words block).

    Strategy: among the top_words, the highest-frequency terms that are NOT
    in BRIDGE_CONCEPT_WHITELIST are confirmed corpus noise.  We keep the top
    top_words_keep_pct fraction as content-bearing and flag the rest.

    Additionally always strips homework / structural section terms confirmed
    as noise by the query_intent EDA breakdown.

    Parameters
    ----------
    eda_stats           : parsed EDA JSON dict (must contain "top_words" key)
    top_words_keep_pct  : fraction of top_words to treat as meaningful [0, 1]
                          0.4 -> keep top 40% by frequency, drop the rest

    Returns
    -------
    set[str] of stopword strings, excluding BRIDGE_CONCEPT_WHITELIST
    """
    try:
        
        if "top_words" not in eda_stats:
            logger.info(f"[eda_derived_stopwords] Received eda_stats with {len(eda_stats)} keys")
            logger.info(f"[eda_derived_stopwords] eda_stats keys: {list(eda_stats.keys()) if eda_stats else 'EMPTY DICT'}")
            raise KeyError(
                f"'top_words' key not found in eda_stats. "
                f"Available keys: {list(eda_stats.keys())}"
            )
        if not (0.0 <= top_words_keep_pct <= 1.0):
            raise ValueError(
                f"top_words_keep_pct must be in [0, 1], got {top_words_keep_pct}"
            )

        top_words: dict[str, int] = eda_stats["top_words"]
        if not top_words:
            logger.warning("[eda_derived_stopwords] top_words block is empty.")
            return set()

        sorted_words = sorted(top_words.items(), key=lambda x: -x[1])
        n_keep       = max(1, int(len(sorted_words) * top_words_keep_pct))

        frequency_stops = {
            word
            for word, _ in sorted_words[n_keep:]
            if word not in BRIDGE_CONCEPT_WHITELIST
        }

        # Confirmed by query_intent EDA: these appear in questions but carry
        # no discriminative signal for retrieval
        structural_noise = {
            "homework", "course", "module", "project", "running",
            "run", "find", "set", "start", "create", "code",
            "after", "was", "your", "failed", "error",
        } - BRIDGE_CONCEPT_WHITELIST

        stopwords = frequency_stops | structural_noise

        logger.info(
            f"[eda_derived_stopwords] "
            f"top_words_keep_pct={top_words_keep_pct} "
            f"-> {len(frequency_stops)} frequency stops "
            f"+ {len(structural_noise)} structural-noise terms "
            f"= {len(stopwords)} total"
        )
        logger.debug(f"Sample: {sorted(stopwords)[:20]}")
        return stopwords

    except Exception:
        logger.error(
            "[eda_derived_stopwords] Failed.\n" + traceback.format_exc()
        )
        raise


# ─────────────────────────────────────────────────────────────────────────────
# IMBALANCE — course-aware corpus balancing
# ─────────────────────────────────────────────────────────────────────────────
def balance_corpus(
    df: pd.DataFrame,
    course_col: str = "course",
    strategy: str = "oversample_min",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Address the 6:1 course imbalance confirmed by EDA before TF-IDF fitting
    so that per-course scores are comparable.

    Strategies
    ----------
    oversample_min   : duplicate minority-course chunks to match the median
                       course size.  Preserves all data, inflates llm-zoomcamp.
    undersample_max  : sample majority courses down to the median size.
                       Loses data but produces cleaner per-course TF-IDF.
    none             : return df unchanged (caller accepts imbalanced scores)

    Parameters
    ----------
    df              : chunk DataFrame
    course_col      : column holding course identifiers
    strategy        : one of "oversample_min", "undersample_max", "none"
    random_state    : numpy/pandas random seed for reproducibility

    Returns
    -------
    Balanced DataFrame (index reset)
    """
    try:
        if course_col not in df.columns:
            raise ValueError(
                f"Column '{course_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        counts  = df[course_col].value_counts()
        median  = int(counts.median())

        logger.info(
            f"[balance_corpus] strategy={strategy}, "
            f"course counts before: {counts.to_dict()}, target={median}"
        )

        if strategy == "none":
            return df.copy()

        rng    = np.random.default_rng(random_state)
        frames = []

        for course, group in df.groupby(course_col):
            n = len(group)
            if strategy == "oversample_min":
                if n < median:
                    extra_idx = rng.choice(group.index, size=median - n, replace=True)
                    frames.append(pd.concat([group, df.loc[extra_idx]]))
                else:
                    frames.append(group)
            elif strategy == "undersample_max":
                if n > median:
                    frames.append(group.sample(n=median, random_state=random_state))
                else:
                    frames.append(group)
            else:
                raise ValueError(
                    f"Unknown strategy '{strategy}'. "
                    f"Choose from: oversample_min, undersample_max, none"
                )

        result = pd.concat(frames).reset_index(drop=True)

        logger.info(
            f"[balance_corpus] course counts after: "
            f"{result[course_col].value_counts().to_dict()}"
        )
        return result

    except Exception:
        logger.error("[balance_corpus] Failed.\n" + traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING — code block stripping
# ─────────────────────────────────────────────────────────────────────────────
def strip_code_blocks(text: str) -> str:
    """
    Remove fenced code blocks before TF-IDF fitting.

    652/1204 chunks (54%) contain code blocks per EDA.  Without stripping,
    code tokens (docker run, pip install, import x) inflate term frequencies
    in ways that don't reflect natural language meaning.

    Parameters
    ----------
    text    : raw chunk text

    Returns
    -------
    text with fenced code blocks replaced by a single space
    """
    try:
        import re
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}")
        # Remove triple-backtick blocks (with optional language tag)
        cleaned = re.sub(r"```[\s\S]*?```", " ", text)
        # Remove inline backtick spans
        cleaned = re.sub(r"`[^`]+`", " ", cleaned)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    except Exception:
        logger.error("[strip_code_blocks] Failed.\n" + traceback.format_exc())
        raise


def apply_code_stripping(
    df: pd.DataFrame,
    text_col: str = "clean_text",
    output_col: str = "tfidf_text",
) -> pd.DataFrame:
    """
    Apply strip_code_blocks across the corpus, writing to a new column so
    the original clean_text is preserved for embedding / other uses.

    Parameters
    ----------
    df          : chunk DataFrame
    text_col    : source column
    output_col  : destination column (created or overwritten)

    Returns
    -------
    DataFrame with output_col added
    """
    try:
        if text_col not in df.columns:
            raise ValueError(
                f"Column '{text_col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

        df = df.copy()
        df[output_col] = df[text_col].map(strip_code_blocks)

        n_changed = (df[output_col] != df[text_col]).sum()
        logger.info(
            f"[apply_code_stripping] "
            f"{text_col} -> {output_col}: "
            f"{n_changed}/{len(df)} chunks modified"
        )
        return df

    except Exception:
        logger.error("[apply_code_stripping] Failed.\n" + traceback.format_exc())
        raise


# ─────────────────────────────────────────────────────────────────────────────
# MASTER BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def build_stopword_list(
    df: pd.DataFrame | None = None,
    eda_stats: dict | None = None,
    tfidf_results: dict | None = None,
    vectorizer: TfidfVectorizer | None = None,
    extra: set[str] | None = None,
) -> list[str]:
    """
    Build a stopword list entirely from data.  Each argument unlocks an
    additional pass — call with progressively more information as fitting
    proceeds.

    Pass availability
    -----------------
    Always          : sklearn ENGLISH_STOP_WORDS
    df              : + frequency_based_stopwords()
    eda_stats       : + eda_derived_stopwords()
    tfidf_results   : + saturation_based_stopwords()
    vectorizer      : + idf_floor_stopwords()
    extra           : + caller-supplied terms

    BRIDGE_CONCEPT_WHITELIST is enforced in every individual pass function
    and again as a final safety net here.

    Parameters
    ----------
    df              : chunk DataFrame
    eda_stats       : parsed EDA JSON dict
    tfidf_results   : {course: [(term, score), ...]}
    vectorizer      : fitted TfidfVectorizer
    extra           : additional stopwords to merge in

    Returns
    -------
    Sorted list[str] — sorted for diff-friendly versioning
    """
    try:
        stops: set[str] = set(ENGLISH_STOP_WORDS)
        logger.info(f"[build_stopword_list] Base sklearn stops: {len(stops)}")

        if df is not None:
            stops |= frequency_based_stopwords(df)
            logger.info(f"[build_stopword_list] After frequency pass: {len(stops)}")

        if eda_stats is not None:
            stops |= eda_derived_stopwords(eda_stats)
            logger.info(f"[build_stopword_list] After EDA pass: {len(stops)}")

        if tfidf_results is not None:
            stops |= saturation_based_stopwords(tfidf_results)
            logger.info(f"[build_stopword_list] After saturation pass: {len(stops)}")

        if vectorizer is not None:
            stops |= idf_floor_stopwords(vectorizer)
            logger.info(f"[build_stopword_list] After IDF-floor pass: {len(stops)}")

        if extra:
            protected = extra & BRIDGE_CONCEPT_WHITELIST
            if protected:
                logger.warning(
                    f"[build_stopword_list] Ignoring {len(protected)} whitelisted "
                    f"terms passed in extra: {sorted(protected)}"
                )
            stops |= extra - BRIDGE_CONCEPT_WHITELIST
            logger.info(f"[build_stopword_list] After extra pass: {len(stops)}")

        # Final safety — whitelist always wins
        stops -= BRIDGE_CONCEPT_WHITELIST

        result = sorted(stops)
        logger.info(f"[build_stopword_list] Final stopword list: {len(result)} terms")
        return result

    except Exception:
        logger.error(
            "[build_stopword_list] Failed.\n" + traceback.format_exc()
        )
        raise


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────
def save_stopwords(
    stopwords: list[str],
    output_dir: Path,
    filename: str = "stopwords.txt",
) -> Path:
    """
    Save stopword list to disk for versioning and reproducibility.

    Parameters
    ----------
    stopwords   : list of stopword strings
    output_dir  : directory to write to (created if absent)
    filename    : output filename

    Returns
    -------
    Path to written file
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / filename
        out_path.write_text("\n".join(stopwords), encoding="utf-8")
        logger.info(f"[save_stopwords] {len(stopwords)} terms -> {out_path}")
        return out_path

    except Exception:
        logger.error("[save_stopwords] Failed.\n" + traceback.format_exc())
        raise


def load_stopwords(path: Path) -> list[str]:
    """
    Load a previously saved stopword list from disk.

    Parameters
    ----------
    path    : path to the stopwords .txt file

    Returns
    -------
    list[str]
    """
    try:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Stopword file not found: {path}")
        words = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        logger.info(f"[load_stopwords] {len(words)} terms <- {path}")
        return words

    except Exception:
        logger.error("[load_stopwords] Failed.\n" + traceback.format_exc())
        raise