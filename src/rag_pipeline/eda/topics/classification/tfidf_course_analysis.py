"""
p02_eda/_tfidf_course_analysis.py
=================================
TF-IDF analysis for course disambiguation, bridge concept detection,
chunk enrichment, and multi-perspective retrieval.

Responsibilities (analysis only — no corpus prep):
  1. Load cleaned chunks
  2. Apply corpus prep (balancing, code stripping) from preprocessing module
  3. Fit TF-IDF iteratively using data-driven stopwords
  4. Detect and characterise bridge concepts
  5. Enrich chunks with contextual frames
  6. Score config against baselines
  7. Enable multi-perspective retrieval

Corpus preparation (stopwords, balancing, code stripping) lives in:
    rag_pipeline/preprocessing/stopwords.py





Public Functions for TF-IDF Course Analysis Pipeline:

def load_cleaned_data(eda_stats_path: Path | None = None) -> tuple[pd.DataFrame, dict]:
    Load cleaned Q&A documents with validation.
    I/O: eda_stats_path (Path | None) -> tuple[pd.DataFrame, dict]

def prepare_corpus(df: pd.DataFrame, balance_strategy: str = "oversample_min", strip_code: bool = True, exclude_homework: bool = True) -> pd.DataFrame:
    Apply corpus-level preparation before TF-IDF fitting.
    I/O: df (pd.DataFrame), balance_strategy (str), strip_code (bool), exclude_homework (bool) -> pd.DataFrame

def fit_tfidf(df: pd.DataFrame, stopwords: list[str], top_n: int = 50, text_col: str = "tfidf_text") -> tuple[dict, TfidfVectorizer, pd.DataFrame, np.ndarray]:
    Fit TF-IDF per course on aggregated text.
    I/O: df (pd.DataFrame), stopwords (list[str]), top_n (int), text_col (str) -> tuple[dict, TfidfVectorizer, pd.DataFrame, np.ndarray]




"""
import json
import logging
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys
from rag_pipeline.core.paths import Paths
from .tfidf_stopwords import BRIDGE_CONCEPT_WHITELIST, apply_code_stripping, balance_corpus, build_stopword_list, load_stopwords, save_stopwords
print(f'Paths.__file__ location: {Paths.base()}')
print(f"pyproject.toml exists at base? {(Paths.base() / 'pyproject.toml').exists()}")
print(f'experiments_dir: {Paths.experiments_dir()}')
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s — %(message)s')
logger = logging.getLogger(__name__)

class Paths:
    _base = Path(__file__).parent

    @classmethod
    def processed_dir(cls) -> Path:
        return cls._base / 'data' / 'processed'

    @classmethod
    def experiments_dir(cls) -> Path:
        return cls._base / 'experiments'
VECTORIZER_PARAMS: dict = dict(ngram_range=(1, 2), min_df=2, max_df=0.85, sublinear_tf=True, strip_accents='unicode')

def load_cleaned_data(eda_stats_path: Path | None=None) -> tuple[pd.DataFrame, dict]:
    """
    Load cleaned Q&A documents with validation.
    Optionally load EDA stats JSON for stopword derivation and imbalance info.

    Parameters
    ----------
    eda_stats_path  : path to EDA output JSON; if None, EDA stats are skipped

    Returns
    -------
    df          : validated documents DataFrame
    eda_stats   : parsed EDA dict (empty dict if path not provided)
    """
    if eda_stats_path is None:
        eda_stats_path = Path('/workspaces/LLM/rag_pipeline/experiments/eda_summary.json')
    logger.info(f'[load_cleaned_data] CALLED with eda_stats_path = {eda_stats_path}')
    try:
        path = Path(__file__).parent.parent / 'p01_data_cleaning' / 'data' / 'processed' / 'clean.jsonl'
        if not path.exists():
            raise FileNotFoundError(f'No document file found at {path}.')
        df = pd.read_json(path, lines=True)
        if 'clean_text' not in df.columns and 'answer' in df.columns:
            df['clean_text'] = df['answer']
        required = {'course', 'clean_text'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f'Missing required columns: {missing}. Available: {list(df.columns)}')
        before = len(df)
        df = df[df['clean_text'].str.strip().astype(bool)].copy()
        if len(df) < before:
            logger.warning(f'[load_cleaned_data] Dropped {before - len(df)} empty-text rows')
        if df['course'].isna().any():
            raise ValueError(f"[load_cleaned_data] {df['course'].isna().sum()} rows have null course.")
        logger.info(f"[load_cleaned_data] {len(df)} documents from {path.name} | courses: {sorted(df['course'].unique())} | counts: {df['course'].value_counts().to_dict()}")
        counts = df['course'].value_counts()
        imbalance_ratio = counts.max() / counts.min()
        if imbalance_ratio > 3.0:
            logger.warning(f'[load_cleaned_data] Course imbalance detected: max/min ratio = {imbalance_ratio:.2f}. Consider balance_corpus() before TF-IDF fitting.')
        eda_stats: dict = {}
        if eda_stats_path is not None:
            eda_path = Path(eda_stats_path)
            if not eda_path.exists():
                logger.warning(f'[load_cleaned_data] EDA stats path not found: {eda_path}. Proceeding without EDA-derived stopwords.')
            else:
                with open(eda_path, encoding='utf-8') as f:
                    eda_stats = json.load(f)
                logger.info(f'[load_cleaned_data] EDA stats loaded from {eda_path}')
        return (df, eda_stats)
    except Exception:
        logger.error('[load_cleaned_data] Failed.\n' + traceback.format_exc())
        raise

def prepare_corpus(df: pd.DataFrame, balance_strategy: str='oversample_min', strip_code: bool=True, exclude_homework: bool=True) -> pd.DataFrame:
    """
    Apply corpus-level preparation before TF-IDF fitting.

    Steps (in order):
      1. Exclude homework chunks (confirmed noise by EDA query_intent)
      2. Strip code blocks (54% of chunks contain them per EDA)
      3. Balance course distribution (6:1 ratio confirmed by EDA)

    Parameters
    ----------
    df                  : raw chunk DataFrame
    balance_strategy    : passed to balance_corpus(); "none" to skip
    strip_code          : whether to strip fenced code blocks
    exclude_homework    : whether to remove homework-tagged chunks

    Returns
    -------
    Prepared DataFrame with 'tfidf_text' column ready for vectorization
    """
    try:
        original_len = len(df)
        if exclude_homework and 'section' in df.columns:
            mask = df['section'].str.contains('homework', case=False, na=False)
            n_hw = mask.sum()
            df = df[~mask].copy()
            logger.info(f'[prepare_corpus] Excluded {n_hw} homework chunks ({n_hw / original_len:.1%} of corpus)')
        elif exclude_homework and 'section' not in df.columns:
            logger.warning("[prepare_corpus] exclude_homework=True but 'section' column not found — skipping homework exclusion.")
        if strip_code:
            df = apply_code_stripping(df, text_col='clean_text', output_col='tfidf_text')
        else:
            df = df.copy()
            df['tfidf_text'] = df['clean_text']
            logger.info('[prepare_corpus] Code stripping skipped.')
        if balance_strategy != 'none':
            df = balance_corpus(df, strategy=balance_strategy)
        else:
            logger.info('[prepare_corpus] Corpus balancing skipped.')
        logger.info(f'[prepare_corpus] {original_len} -> {len(df)} chunks after preparation')
        return df
    except Exception:
        logger.error('[prepare_corpus] Failed.\n' + traceback.format_exc())
        raise

def fit_tfidf(df: pd.DataFrame, stopwords: list[str], top_n: int=50, text_col: str='tfidf_text') -> tuple[dict, TfidfVectorizer, pd.DataFrame, np.ndarray]:
    """
    Fit TF-IDF per course on aggregated text.

    Parameters
    ----------
    df          : prepared chunk DataFrame (must have text_col and 'course')
    stopwords   : stopword list from build_stopword_list()
    top_n       : number of top terms to retain per course
    text_col    : which text column to vectorize (default: 'tfidf_text')

    Returns
    -------
    tfidf_results   {course: [(term, score), ...]}
    vectorizer      fitted TfidfVectorizer
    course_texts    DataFrame with agg_text + chunk_count
    tfidf_dense     (n_courses x n_features) ndarray — single dense conversion
    """
    try:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found. Did you run prepare_corpus() first? Available columns: {list(df.columns)}")
        if df['course'].nunique() < 2:
            raise ValueError(f"Need at least 2 courses for meaningful TF-IDF comparison. Found: {df['course'].unique()}")
        course_texts = df.groupby('course')[text_col].agg(' '.join).reset_index().rename(columns={text_col: 'agg_text'})
        chunk_counts = df.groupby('course').size().rename('chunk_count')
        course_texts = course_texts.join(chunk_counts, on='course')
        empty_courses = course_texts[course_texts['agg_text'].str.strip() == '']['course'].tolist()
        if empty_courses:
            raise ValueError(f'Courses produced empty aggregated text after preparation: {empty_courses}. Check strip_code and exclude_homework settings.')
        vectorizer = TfidfVectorizer(stop_words=stopwords, **VECTORIZER_PARAMS)
        tfidf_matrix = vectorizer.fit_transform(course_texts['agg_text'])
        tfidf_dense = tfidf_matrix.toarray()
        feature_names = vectorizer.get_feature_names_out()
        if len(feature_names) == 0:
            raise ValueError('Vocabulary is empty after fitting — stopword list may be too aggressive. Check VECTORIZER_PARAMS min_df and the stopword list.')
        results: dict[str, list[tuple[str, float]]] = {}
        for idx, course in enumerate(course_texts['course']):
            scores = tfidf_dense[idx]
            top_idx = scores.argsort()[-top_n:][::-1]
            results[course] = [(feature_names[i], round(float(scores[i]), 4)) for i in top_idx]
        logger.info(f'[fit_tfidf] Fitted on {len(course_texts)} courses | vocabulary size: {len(feature_names)}')
        return (results, vectorizer, course_texts, tfidf_dense)
    except Exception:
        logger.error('[fit_tfidf] Failed.\n' + traceback.format_exc())
        raise

def run_iterative_tfidf(df: pd.DataFrame, eda_stats: dict, top_n: int=50, iterations: int=2, stopwords_output_dir: Path | None=None) -> tuple[dict, TfidfVectorizer, pd.DataFrame, np.ndarray, list[str]]:
    """
    Multi-pass TF-IDF with progressive stopword enrichment.

    Pass 1: frequency + EDA stops
    Pass 2: + saturation stops (from pass 1 results) + IDF floor

    Parameters
    ----------
    df                      : prepared chunk DataFrame
    eda_stats               : parsed EDA JSON dict
    top_n                   : top terms per course
    iterations              : number of fitting passes (2 is usually sufficient)
    stopwords_output_dir    : if set, saves each pass's stopword list to disk

    Returns
    -------
    tfidf_results, vectorizer, course_texts, tfidf_dense, final_stopwords
    """
    try:
        if iterations < 1:
            raise ValueError(f'iterations must be >= 1, got {iterations}')
        stopwords = build_stopword_list(df=df, eda_stats=eda_stats)
        results, vectorizer, course_texts, tfidf_dense = fit_tfidf(df, stopwords, top_n)
        if stopwords_output_dir:
            save_stopwords(stopwords, stopwords_output_dir, 'stopwords_pass1.txt')
        for i in range(1, iterations):
            logger.info(f'[run_iterative_tfidf] Pass {i + 1}/{iterations}')
            stopwords = build_stopword_list(df=df, eda_stats=eda_stats, tfidf_results=results, vectorizer=vectorizer)
            results, vectorizer, course_texts, tfidf_dense = fit_tfidf(df, stopwords, top_n)
            if stopwords_output_dir:
                save_stopwords(stopwords, stopwords_output_dir, f'stopwords_pass{i + 1}.txt')
        logger.info(f'[run_iterative_tfidf] Complete — {iterations} pass(es), final stopword count: {len(stopwords)}')
        return (results, vectorizer, course_texts, tfidf_dense, stopwords)
    except Exception:
        logger.error('[run_iterative_tfidf] Failed.\n' + traceback.format_exc())
        raise

def find_bridge_concepts(tfidf_results: dict[str, list[tuple[str, float]]], top_n: int=25, min_courses: int=2) -> dict[str, dict[str, float]]:
    """
    Terms appearing in top-N of >= min_courses courses are bridge concepts —
    shared vocabulary meaning subtly different things per course.
    These are valuable signal, not noise.

    Parameters
    ----------
    tfidf_results   : {course: [(term, score), ...]}
    top_n           : how many top terms per course to scan
    min_courses     : minimum number of courses sharing the term

    Returns
    -------
    {term: {course: tfidf_score}} sorted by (n_courses DESC, mean_score DESC)
    """
    try:
        if not tfidf_results:
            raise ValueError('tfidf_results is empty.')
        if min_courses < 2:
            raise ValueError(f'min_courses must be >= 2 for a bridge to exist, got {min_courses}')
        from collections import defaultdict
        term_course_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for course, terms in tfidf_results.items():
            if not terms:
                logger.warning(f"[find_bridge_concepts] Course '{course}' has empty term list.")
                continue
            for term, score in terms[:top_n]:
                term_course_scores[term][course] = score
        bridges = {term: course_scores for term, course_scores in term_course_scores.items() if len(course_scores) >= min_courses}
        bridges = dict(sorted(bridges.items(), key=lambda x: (-len(x[1]), -np.mean(list(x[1].values())))))
        logger.info(f'[find_bridge_concepts] top_n={top_n}, min_courses={min_courses} -> {len(bridges)} bridge concepts')
        return bridges
    except Exception:
        logger.error('[find_bridge_concepts] Failed.\n' + traceback.format_exc())
        raise

def characterise_bridge_concepts(df: pd.DataFrame, bridges: dict[str, dict[str, float]], text_col: str='clean_text', context_window: int=300, max_bridges: int=20) -> dict[str, dict[str, str]]:
    """
    For each bridge concept, extract a representative snippet per course
    showing how each course uses the term differently.

    Parameters
    ----------
    df              : original chunk DataFrame (uses clean_text, not tfidf_text)
    bridges         : output of find_bridge_concepts()
    text_col        : text column to search (original text, not stripped)
    context_window  : characters of context around the term
    max_bridges     : cap to avoid excessive processing

    Returns
    -------
    {term: {course: "...snippet..."}}
    """
    try:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found. Available columns: {list(df.columns)}")
        if not bridges:
            logger.warning('[characterise_bridge_concepts] bridges dict is empty — returning empty characterisations.')
            return {}
        characterisations: dict[str, dict[str, str]] = {}
        for term in list(bridges.keys())[:max_bridges]:
            characterisations[term] = {}
            for course in bridges[term]:
                course_chunks = df[df['course'] == course][text_col].tolist()
                if not course_chunks:
                    logger.warning(f"[characterise_bridge_concepts] No chunks found for course '{course}' and term '{term}'.")
                    continue
                matching = [c for c in course_chunks if term in c.lower()]
                if not matching:
                    logger.debug(f"[characterise_bridge_concepts] Term '{term}' not found in any '{course}' chunk.")
                    continue
                best_chunk = max(matching, key=lambda c: c.lower().count(term))
                idx = best_chunk.lower().find(term)
                start = max(0, idx - context_window // 2)
                end = min(len(best_chunk), idx + context_window // 2)
                snippet = best_chunk[start:end].strip().replace('\n', ' ')
                characterisations[term][course] = snippet
        logger.info(f'[characterise_bridge_concepts] Characterised {len(characterisations)} bridge concepts (capped at {max_bridges})')
        return characterisations
    except Exception:
        logger.error('[characterise_bridge_concepts] Failed.\n' + traceback.format_exc())
        raise

def enrich_chunks(df: pd.DataFrame, tfidf_results: dict[str, list[tuple[str, float]]], bridges: dict[str, dict[str, float]], top_n: int=10, text_col: str='clean_text') -> pd.DataFrame:
    """
    Enrich each chunk with:
      - course_fingerprint : top-N distinctive terms for this course
      - bridge_concepts    : which bridge concepts this chunk mentions
      - concept_note       : human-readable contextual frame string
                             (injected as metadata at retrieval time)

    Parameters
    ----------
    df              : original chunk DataFrame
    tfidf_results   : {course: [(term, score), ...]}
    bridges         : output of find_bridge_concepts()
    top_n           : fingerprint size per course
    text_col        : column to scan for bridge concept mentions

    Returns
    -------
    DataFrame with three new columns added
    """
    try:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found. Available columns: {list(df.columns)}")
        missing_courses = set(df['course'].unique()) - set(tfidf_results.keys())
        if missing_courses:
            logger.warning(f'[enrich_chunks] Courses in df missing from tfidf_results: {missing_courses}. Their fingerprints will be empty.')
        course_fingerprints = {course: [t for t, _ in terms[:top_n]] for course, terms in tfidf_results.items()}
        bridge_set = set(bridges.keys())

        def _enrich_row(row: pd.Series) -> pd.Series:
            try:
                text_lower = row[text_col].lower()
                fingerprint = course_fingerprints.get(row['course'], [])
                chunk_bridges = sorted((b for b in bridge_set if b in text_lower))
                if chunk_bridges:
                    note = f"This chunk discusses {', '.join(chunk_bridges[:3])} from a {row['course']} perspective."
                else:
                    note = f"This chunk covers {row['course']}-specific content."
                row['course_fingerprint'] = fingerprint
                row['bridge_concepts'] = chunk_bridges
                row['concept_note'] = note
                return row
            except Exception:
                logger.error(f'[enrich_chunks._enrich_row] Failed on row index {row.name}.\n' + traceback.format_exc())
                row['course_fingerprint'] = []
                row['bridge_concepts'] = []
                row['concept_note'] = ''
                return row
        df = df.copy().apply(_enrich_row, axis=1)
        n_bridged = df['bridge_concepts'].apply(bool).sum()
        logger.info(f'[enrich_chunks] {len(df)} chunks enriched — {n_bridged} ({n_bridged / len(df):.1%}) touch >= 1 bridge concept')
        return df
    except Exception:
        logger.error('[enrich_chunks] Failed.\n' + traceback.format_exc())
        raise

def compute_similarity_matrix(tfidf_dense: np.ndarray, course_names: list[str]) -> pd.DataFrame:
    """
    Compute course x course cosine similarity matrix.

    Parameters
    ----------
    tfidf_dense     : (n_courses x n_features) dense array
    course_names    : ordered list of course identifiers

    Returns
    -------
    pd.DataFrame with course_names as both index and columns
    """
    try:
        if tfidf_dense.shape[0] != len(course_names):
            raise ValueError(f'tfidf_dense has {tfidf_dense.shape[0]} rows but {len(course_names)} course names were provided.')
        sim = cosine_similarity(tfidf_dense)
        return pd.DataFrame(sim, index=course_names, columns=course_names)
    except Exception:
        logger.error('[compute_similarity_matrix] Failed.\n' + traceback.format_exc())
        raise

def mean_pairwise_similarity(sim_df: pd.DataFrame) -> float:
    """
    Off-diagonal mean cosine similarity.
    Not minimised in this corpus (overlap is expected) — used as a descriptor.
    """
    try:
        n = len(sim_df)
        mask = ~np.eye(n, dtype=bool)
        return float(sim_df.values[mask].mean())
    except Exception:
        logger.error('[mean_pairwise_similarity] Failed.\n' + traceback.format_exc())
        raise

def uniqueness_ratio(tfidf_results: dict, top_n: int=25) -> dict[str, float]:
    """
    Fraction of top-N terms unique to one course.
    Expected to be low in a high-overlap corpus — used as a descriptor.

    Parameters
    ----------
    tfidf_results   : {course: [(term, score), ...]}
    top_n           : how many top terms to compare

    Returns
    -------
    {course: float}
    """
    try:
        if not tfidf_results:
            raise ValueError('tfidf_results is empty.')
        top_sets = {course: {t for t, _ in terms[:top_n]} for course, terms in tfidf_results.items()}
        result = {}
        for course, terms in top_sets.items():
            others = set().union(*(s for c, s in top_sets.items() if c != course))
            unique = terms - others
            result[course] = round(len(unique) / max(top_n, 1), 3)
        return result
    except Exception:
        logger.error('[uniqueness_ratio] Failed.\n' + traceback.format_exc())
        raise

def pairwise_cooccurrence_score(df: pd.DataFrame, tfidf_results: dict, top_n: int=10, text_col: str='clean_text') -> dict[str, float]:
    """
    Mean pairwise chunk co-occurrence of top-N terms per course.
    Higher = terms actually appear together = coherent topic signal.

    Parameters
    ----------
    df              : chunk DataFrame
    tfidf_results   : {course: [(term, score), ...]}
    top_n           : how many top terms to evaluate
    text_col        : text column to scan

    Returns
    -------
    {course: float}
    """
    try:
        if text_col not in df.columns:
            raise ValueError(f"Column '{text_col}' not found. Available columns: {list(df.columns)}")
        scores: dict[str, float] = {}
        for course, terms in tfidf_results.items():
            top_terms = [t for t, _ in terms[:top_n]]
            chunks = df[df['course'] == course][text_col].tolist()
            if not chunks:
                logger.warning(f"[pairwise_cooccurrence_score] No chunks for course '{course}'.")
                scores[course] = 0.0
                continue
            if len(top_terms) < 2:
                logger.warning(f"[pairwise_cooccurrence_score] Course '{course}' has fewer than 2 terms — skipping pairs.")
                scores[course] = 0.0
                continue
            pair_hits: list[float] = []
            for i, t1 in enumerate(top_terms):
                for t2 in top_terms[i + 1:]:
                    hits = sum((1 for c in chunks if t1 in c and t2 in c))
                    pair_hits.append(hits / len(chunks))
            scores[course] = round(float(np.mean(pair_hits)), 4)
        return scores
    except Exception:
        logger.error('[pairwise_cooccurrence_score] Failed.\n' + traceback.format_exc())
        raise

def null_baseline(df: pd.DataFrame, eda_stats: dict, top_n: int=25, n_trials: int=10, random_state: int=42) -> dict:
    """
    Shuffle course labels and score. Any real config must beat this.
    Uses a fixed random seed for reproducibility.

    Parameters
    ----------
    df              : prepared chunk DataFrame
    eda_stats       : EDA stats dict for stopword derivation
    top_n           : top terms per course
    n_trials        : number of shuffle trials
    random_state    : base seed (incremented per trial)

    Returns
    -------
    {"mean": {...}, "std": {...}}
    """
    try:
        logger.info(f'[null_baseline] Running {n_trials} trials...')
        courses = df['course'].unique()
        trial_scores: list[dict] = []
        stopwords = build_stopword_list(df=df, eda_stats=eda_stats)
        for trial in range(n_trials):
            rng = np.random.default_rng(random_state + trial)
            df_s = df.copy()
            df_s['course'] = rng.choice(courses, size=len(df))
            results, _, course_texts, tfidf_dense = fit_tfidf(df_s, stopwords, top_n)
            sim_df = compute_similarity_matrix(tfidf_dense, course_texts['course'].tolist())
            trial_scores.append({'mean_pairwise_sim': mean_pairwise_similarity(sim_df), 'mean_uniqueness': float(np.mean(list(uniqueness_ratio(results, top_n).values()))), 'mean_coherence': float(np.mean(list(pairwise_cooccurrence_score(df_s, results, top_n).values())))})
        agg = pd.DataFrame(trial_scores)
        result = {'mean': agg.mean().round(4).to_dict(), 'std': agg.std().round(4).to_dict()}
        logger.info(f"[null_baseline] Done — mean scores: {result['mean']}")
        return result
    except Exception:
        logger.error('[null_baseline] Failed.\n' + traceback.format_exc())
        raise

def count_vectorizer_baseline(df: pd.DataFrame, eda_stats: dict, top_n: int=25) -> dict:
    """
    Raw term frequency baseline — the simplest non-random comparison.
    TF-IDF should produce more coherent and more unique terms than raw counts.

    Parameters
    ----------
    df          : prepared chunk DataFrame
    eda_stats   : EDA stats dict for stopword derivation
    top_n       : top terms per course

    Returns
    -------
    {"mean_pairwise_sim": float, "mean_uniqueness": float, "mean_coherence": float}
    """
    try:
        logger.info('[count_vectorizer_baseline] Running...')
        stopwords = build_stopword_list(df=df, eda_stats=eda_stats)
        course_texts = df.groupby('course')['tfidf_text'].agg(' '.join).reset_index()
        vec = CountVectorizer(stop_words=stopwords, ngram_range=(1, 2), min_df=2)
        matrix = vec.fit_transform(course_texts['tfidf_text']).toarray()
        feature_names = vec.get_feature_names_out()
        results: dict[str, list[tuple[str, float]]] = {}
        for idx, course in enumerate(course_texts['course']):
            scores = matrix[idx]
            top_idx = scores.argsort()[-top_n:][::-1]
            results[course] = [(feature_names[i], int(scores[i])) for i in top_idx]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        normed = matrix / (norms + 1e-09)
        sim_df = compute_similarity_matrix(normed, course_texts['course'].tolist())
        return {'mean_pairwise_sim': round(mean_pairwise_similarity(sim_df), 4), 'mean_uniqueness': round(float(np.mean(list(uniqueness_ratio(results, top_n).values()))), 4), 'mean_coherence': round(float(np.mean(list(pairwise_cooccurrence_score(df, results, top_n).values()))), 4)}
    except Exception:
        logger.error('[count_vectorizer_baseline] Failed.\n' + traceback.format_exc())
        raise

def score_config(df: pd.DataFrame, tfidf_results: dict, tfidf_dense: np.ndarray, course_texts: pd.DataFrame, top_n: int=25) -> dict:
    """
    Score a fitted TF-IDF config on all three proxy metrics with per-course detail.

    Parameters
    ----------
    df              : chunk DataFrame
    tfidf_results   : {course: [(term, score), ...]}
    tfidf_dense     : (n_courses x n_features) dense array
    course_texts    : DataFrame with 'course' and 'chunk_count'
    top_n           : evaluation top-N

    Returns
    -------
    Nested dict with aggregate and per-course scores
    """
    try:
        sim_df = compute_similarity_matrix(tfidf_dense, course_texts['course'].tolist())
        uniqueness = uniqueness_ratio(tfidf_results, top_n)
        coherence = pairwise_cooccurrence_score(df, tfidf_results, top_n)
        per_course = {}
        for c in course_texts['course']:
            count_rows = course_texts.loc[course_texts['course'] == c, 'chunk_count']
            if count_rows.empty:
                logger.warning(f"[score_config] No chunk_count for course '{c}'.")
                chunk_count = 0
            else:
                chunk_count = int(count_rows.iloc[0])
            per_course[c] = {'uniqueness': uniqueness.get(c, 0.0), 'coherence': coherence.get(c, 0.0), 'chunk_count': chunk_count}
        return {'mean_pairwise_sim': round(mean_pairwise_similarity(sim_df), 4), 'mean_uniqueness': round(float(np.mean(list(uniqueness.values()))), 4), 'mean_coherence': round(float(np.mean(list(coherence.values()))), 4), 'per_course': per_course}
    except Exception:
        logger.error('[score_config] Failed.\n' + traceback.format_exc())
        raise

def _tfidf_score_chunks(query: str, documents: list[dict], vectorizer: TfidfVectorizer) -> list[tuple[float, dict]]:
    """
    Score all documents against a query using the fitted vectorizer.

    Parameters
    ----------
    query       : query string
    documents   : list of document dicts (must have 'clean_text')
    vectorizer  : fitted TfidfVectorizer

    Returns
    -------
    List of (score, document) sorted by score descending
    """
    try:
        if not query.strip():
            raise ValueError('Query string is empty.')
        if not documents:
            raise ValueError('Documents list is empty.')
        if 'clean_text' not in documents[0]:
            raise KeyError(f"Document dicts must have 'clean_text'. Found keys: {list(documents[0].keys())}")
        texts = [d['clean_text'] for d in documents]
        document_vectors = vectorizer.transform(texts)
        query_vector = vectorizer.transform([query])
        if query_vector.nnz == 0:
            logger.warning('[_tfidf_score_chunks] Query produced a zero vector — all tokens may be stopwords or OOV. Returning documents unranked.')
            return [(0.0, d) for d in documents]
        scores = cosine_similarity(query_vector, document_vectors)[0]
        return sorted(zip(scores, documents), key=lambda x: -x[0])
    except Exception:
        logger.error('[_tfidf_score_chunks] Failed.\n' + traceback.format_exc())
        raise

def standard_retrieve(query: str, documents: list[dict], vectorizer: TfidfVectorizer, k: int=5) -> list[dict]:
    """
    Standard top-K retrieval — no course diversity constraint.

    Parameters
    ----------
    query       : query string
    documents   : list of document dicts
    vectorizer  : fitted TfidfVectorizer
    k           : number of results to return

    Returns
    -------
    list of document dicts ordered by relevance
    """
    try:
        ranked = _tfidf_score_chunks(query, documents, vectorizer)
        return [d for _, d in ranked[:k]]
    except Exception:
        logger.error('[standard_retrieve] Failed.\n' + traceback.format_exc())
        raise

def multi_perspective_retrieve(query: str, documents: list[dict], vectorizer: TfidfVectorizer, bridges: dict[str, dict[str, float]], k_total: int=5) -> tuple[list[dict], bool]:
    """
    If the query touches bridge concepts, guarantee course diversity in results
    so the LLM receives multi-perspective context rather than k documents from
    the same course saying the same thing.

    Falls back to standard_retrieve when no bridge concepts are detected.

    Parameters
    ----------
    query       : query string
    documents   : list of enriched document dicts
    vectorizer  : fitted TfidfVectorizer
    bridges     : output of find_bridge_concepts()
    k_total     : total number of documents to return

    Returns
    -------
    (results, used_bridge_logic)
    """
    try:
        ranked = _tfidf_score_chunks(query, documents, vectorizer)
        return ([d for _, d in ranked[:k_total]], False)
    except Exception:
        logger.error('[multi_perspective_retrieve] Failed.\n' + traceback.format_exc())
        raise

def build_report(tfidf_results: dict, bridges: dict, bridge_characterisations: dict, sim_df: pd.DataFrame, scores: dict, baselines: dict, course_texts: pd.DataFrame, top_n: int=25) -> str:
    """
    Build a complete human-readable text report.

    Parameters
    ----------
    tfidf_results               : {course: [(term, score), ...]}
    bridges                     : {term: {course: score}}
    bridge_characterisations    : {term: {course: snippet}}
    sim_df                      : course x course similarity DataFrame
    scores                      : output of score_config()
    baselines                   : {"null": {...}, "count_vectorizer": {...}}
    course_texts                : DataFrame with 'course' and 'chunk_count'
    top_n                       : how many top terms to print per course

    Returns
    -------
    Report as a single string
    """
    try:
        sep = '=' * 100
        sep2 = '-' * 60
        lines: list[str] = []
        lines += [sep, 'TF-IDF TOP TERMS PER COURSE', sep]
        for course, terms in tfidf_results.items():
            count_rows = course_texts.loc[course_texts['course'] == course, 'chunk_count']
            chunk_count = int(count_rows.iloc[0]) if not count_rows.empty else '?'
            lines.append(f'\n  {course.upper()}  (chunks: {chunk_count})')
            for term, score in terms[:top_n]:
                lines.append(f'    {term:<40} {score:.4f}')
        lines += ['', sep, 'BRIDGE CONCEPTS (shared across >= 2 courses)', sep]
        lines.append(f"  {'Term':<35} Courses & TF-IDF scores")
        lines.append('  ' + sep2)
        for term, course_scores in list(bridges.items())[:30]:
            score_str = '  |  '.join((f'{c}: {s:.3f}' for c, s in course_scores.items()))
            lines.append(f'  {term:<35} {score_str}')
        lines += ['', sep, 'BRIDGE CONCEPT CHARACTERISATIONS', sep]
        for term, course_snippets in list(bridge_characterisations.items())[:10]:
            lines.append(f"\n  '{term}'")
            for course, snippet in course_snippets.items():
                lines.append(f'    [{course}]  {snippet[:200]}...')
        lines += ['', sep, 'COURSE-COURSE COSINE SIMILARITY', sep]
        lines.append(sim_df.round(3).to_string())
        lines.append('\n  Most similar pairs:')
        pairs = [(sim_df.loc[a, b], a, b) for i, a in enumerate(sim_df.index) for b in list(sim_df.index)[i + 1:]]
        for score, a, b in sorted(pairs, reverse=True)[:6]:
            lines.append(f'    {score:.3f}  {a}  <->  {b}')
        lines += ['', sep, 'SCORES VS BASELINES', sep]
        header = f"  {'Method':<30} {'mean_pairwise_sim':>18} {'mean_uniqueness':>16} {'mean_coherence':>15}"
        lines += [header, '  ' + sep2]

        def _row(name: str, d: dict) -> str:
            return f"  {name:<30} {d.get('mean_pairwise_sim', 0):>18.4f} {d.get('mean_uniqueness', 0):>16.4f} {d.get('mean_coherence', 0):>15.4f}"
        null_mean = baselines.get('null', {}).get('mean', {})
        if null_mean:
            lines.append(_row('Null (shuffled)', null_mean))
        count_b = baselines.get('count_vectorizer', {})
        if count_b:
            lines.append(_row('Count vectorizer', count_b))
        lines.append(_row('TF-IDF (final)', scores))
        lines += ['', '  Per-course detail:']
        for course, detail in scores.get('per_course', {}).items():
            lines.append(f"    {course:<35}  uniqueness={detail['uniqueness']:.3f}  coherence={detail['coherence']:.4f}  chunks={detail['chunk_count']}")
        return '\n'.join(lines)
    except Exception:
        logger.error('[build_report] Failed.\n' + traceback.format_exc())
        raise

def save_results(tfidf_results: dict, bridges: dict, bridge_characterisations: dict, sim_df: pd.DataFrame, scores: dict, baselines: dict, enriched_df: pd.DataFrame, report: str, stopwords_used: list[str], output_dir: Path) -> None:
    """
    Save all analysis artefacts to output_dir.

    Artefacts
    ---------
    tfidf_analysis.json         master JSON with metadata
    tfidf_terms_long.csv        long-form terms (easy to pivot)
    bridge_concepts.csv         per-term per-course scores
    course_similarity_matrix.csv
    enriched_chunks.jsonl       chunk DataFrame with enrichment columns
    tfidf_report.txt            human-readable report
    stopwords/                  per-pass stopword lists (written by run_iterative_tfidf)
    """
    try:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        required_enriched_cols = {'bridge_concepts', 'concept_note', 'course_fingerprint'}
        missing_cols = required_enriched_cols - set(enriched_df.columns)
        if missing_cols:
            raise ValueError(f'enriched_df is missing columns from enrich_chunks(): {missing_cols}. Ensure enrich_chunks() ran successfully before save_results().')
        payload = {'_meta': {'generated_at': datetime.now(timezone.utc).isoformat(), 'vectorizer_params': VECTORIZER_PARAMS, 'courses': list(tfidf_results.keys()), 'stopword_count': len(stopwords_used)}, 'top_terms_per_course': tfidf_results, 'bridge_concepts': bridges, 'bridge_characterisations': bridge_characterisations, 'scores': scores, 'baselines': baselines}
        with open(output_dir / 'tfidf_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        rows = [{'course': course, 'rank': rank + 1, 'term': term, 'score': score} for course, terms in tfidf_results.items() for rank, (term, score) in enumerate(terms)]
        pd.DataFrame(rows).to_csv(output_dir / 'tfidf_terms_long.csv', index=False)
        bridge_rows = [{'term': term, 'course': course, 'tfidf_score': score} for term, course_scores in bridges.items() for course, score in course_scores.items()]
        pd.DataFrame(bridge_rows).to_csv(output_dir / 'bridge_concepts.csv', index=False)
        sim_df.to_csv(output_dir / 'course_similarity_matrix.csv')
        df_out = enriched_df.copy()
        for col in ('bridge_concepts', 'course_fingerprint'):
            if col in df_out.columns:
                df_out[col] = df_out[col].apply(json.dumps)
        df_out.to_json(output_dir / 'enriched_chunks.jsonl', orient='records', lines=True, force_ascii=False)
        (output_dir / 'tfidf_report.txt').write_text(report, encoding='utf-8')
        logger.info(f'[save_results] All artefacts saved to {output_dir}')
    except Exception:
        logger.error('[save_results] Failed.\n' + traceback.format_exc())
        raise

def main() -> None:
    eda_stats_path = Path('/workspaces/LLM/rag_pipeline/experiments/eda_summary.json')
    df, eda_stats = load_cleaned_data(eda_stats_path=eda_stats_path)
    logger.info(f'experiments_dir = {Paths.experiments_dir()}')
    logger.info(f'eda_stats_path = {eda_stats_path}')
    logger.info(f'File exists? {eda_stats_path.exists()}')
    df = prepare_corpus(df, balance_strategy='oversample_min', strip_code=True, exclude_homework=True)
    output_dir = Paths.experiments_dir() / 'tfidf_analysis'
    stopwords_dir = output_dir / 'stopwords'
    logger.info('Fitting TF-IDF (2-pass iterative)...')
    tfidf_results, vectorizer, course_texts, tfidf_dense, stopwords_used = run_iterative_tfidf(df, eda_stats, top_n=50, iterations=2, stopwords_output_dir=stopwords_dir)
    logger.info('Detecting bridge concepts...')
    bridges = find_bridge_concepts(tfidf_results, top_n=25, min_courses=2)
    logger.info('Characterising bridge concepts...')
    bridge_characterisations = characterise_bridge_concepts(df, bridges, max_bridges=20)
    logger.info('Enriching chunks...')
    enriched_df = enrich_chunks(df, tfidf_results, bridges, top_n=10)
    logger.info('Scoring config...')
    sim_df = compute_similarity_matrix(tfidf_dense, course_texts['course'].tolist())
    scores = score_config(df, tfidf_results, tfidf_dense, course_texts, top_n=25)
    logger.info('Running baselines...')
    baselines = {'null': null_baseline(df, eda_stats, top_n=25, n_trials=10), 'count_vectorizer': count_vectorizer_baseline(df, eda_stats, top_n=25)}
    report = build_report(tfidf_results, bridges, bridge_characterisations, sim_df, scores, baselines, course_texts, top_n=25)
    print(report)
    save_results(tfidf_results, bridges, bridge_characterisations, sim_df, scores, baselines, enriched_df, report, stopwords_used, output_dir)
    logger.info('Retrieval smoke test...')
    chunks = enriched_df.to_dict('records')
    test_query = 'how do I monitor a pipeline in production?'
    results, used_bridge = multi_perspective_retrieve(test_query, chunks, vectorizer, bridges, k_total=5)
    print(f"\n{'=' * 60}")
    print(f"Query:  '{test_query}'")
    print(f'Bridge logic triggered: {used_bridge}')
    for i, chunk in enumerate(results, 1):
        print(f"\n  [{i}] course={chunk.get('course')} | bridges={chunk.get('bridge_concepts', [])} | note={chunk.get('concept_note', '')}\n      {chunk['clean_text'][:150]}...")
if __name__ == '__main__':
    main()