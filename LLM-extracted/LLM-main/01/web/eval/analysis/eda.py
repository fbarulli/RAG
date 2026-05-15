import sys
sys.path.insert(0, '/home/admin/LLM/LLM/01/web')

import json
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from eval.eval_set import get_eval_set_from_es
from src.search import CourseRAGManager
from src.config_manager import load_full_config
import os

# Optional: for plots
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

def remove_stopwords(text):
    stopwords = {'the', 'to', 'i', 'in', 'how', 'a', 'is', 'for', 'and', 'not',
                 'of', 'on', 'with', 'it', 'you', 'that', 'this', 'are', 'be',
                 'at', 'by', 'from', 'or', 'as', 'what', 'why', 'when', 'where',
                 'which', 'who', 'whom', 'do', 'does', 'did', 'have', 'has', 'had',
                 'can', 'could', 'will', 'would', 'should', 'may', 'might', 'my', 'your'}
    words = text.lower().split()
    return [w for w in words if w not in stopwords and len(w) > 2]

def classify_question_type(question: str) -> str:
    q_lower = question.lower()
    if re.match(r'^(how|how to|how do|how can|how could)', q_lower):
        return 'how'
    if re.match(r'^(what|what is|what are|what does)', q_lower):
        return 'what'
    if re.match(r'^(why|why does|why is|why did)', q_lower):
        return 'why'
    if re.match(r'^(error|exception|failed|problem|issue)', q_lower):
        return 'error'
    if re.search(r'difference between| vs ', q_lower):
        return 'comparison'
    if re.search(r'can i|could i|is it possible', q_lower):
        return 'feasibility'
    return 'other'

def load_benchmark_results(config_name):
    import json
    path = f'experiments/results/{config_name}.json'
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        return {r['query']: r for r in data['results'] if r['k'] == 5}
    except:
        return {}

def get_embedding_manager():
    # Load default config to get embedding model
    settings = load_full_config('bm25_default')
    manager = CourseRAGManager(settings)
    manager.connect_elasticsearch()
    return manager

def near_duplicate_queries(queries, manager, threshold=0.95):
    """Return pairs of queries with cosine similarity > threshold."""
    if not manager.embed_model:
        print("No embedding model available for near-duplicate detection")
        return []
    embeddings = [manager.embed_model.get_text_embedding(q) for q in queries]
    sim_matrix = cosine_similarity(embeddings)
    n = len(queries)
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if sim_matrix[i][j] > threshold:
                pairs.append((queries[i], queries[j], sim_matrix[i][j]))
    return pairs

def analyze_dataset():
    eval_set = get_eval_set_from_es()
    queries = []
    courses = []
    question_lengths = []
    word_counts = []
    unique_words_per_query = []
    question_types = []
    # For technical terms: count queries containing each term (not occurrences)
    tech_term_queries = defaultdict(set)
    technical_patterns = {
        'Kafka': r'\bkafka\b',
        'Docker': r'\bdocker\b',
        'Terraform': r'\bterraform\b',
        'AWS': r'\baws\b',
        'GCP': r'\bgcp\b|google cloud',
        'BigQuery': r'\bbigquery\b',
        'PySpark': r'\bpyspark\b',
        'MLflow': r'\bmlflow\b',
        'WandB': r'\bwandb\b',
        'dbt': r'\bdbt\b',
        'PostgreSQL': r'\bpostgres\b',
        'Redis': r'\bredis\b',
        'Kubernetes': r'\bkubernetes\b|k8s',
        'Git': r'\bgit\b',
        'Python': r'\bpython\b',
        'pandas': r'\bpandas\b',
        'numpy': r'\bnumpy\b'
    }

    for item in eval_set:
        doc = item['original_doc']
        question = doc.get('question', '')
        course = doc.get('course', '')
        queries.append(question)
        courses.append(course)
        question_lengths.append(len(question))
        words = question.split()
        word_counts.append(len(words))
        unique_words_per_query.append(len(set(words)))
        qtype = classify_question_type(question)
        question_types.append(qtype)
        # Track technical terms per query (as set)
        for tech, pattern in technical_patterns.items():
            if re.search(pattern, question, re.IGNORECASE):
                tech_term_queries[tech].add(question)

    df = pd.DataFrame({
        'query': queries,
        'course': courses,
        'length': question_lengths,
        'word_count': word_counts,
        'unique_words': unique_words_per_query,
        'question_type': question_types
    })

    print("=== DATASET OVERVIEW ===")
    print(f"Total queries: {len(df)}")
    print(f"Unique courses: {df['course'].nunique()}")
    print("\nCourse distribution (top 10):")
    print(df['course'].value_counts().head(10))

    print("\n=== TECHNICAL TERM DETECTION (queries containing term) ===")
    print(f"Queries with any technical term: {len(set().union(*tech_term_queries.values()))}/{len(df)}")
    print("\nTop technical terms (by number of queries):")
    for tech, qset in sorted(tech_term_queries.items(), key=lambda x: len(x[1]), reverse=True)[:10]:
        print(f"  {tech}: {len(qset)} queries")

    print("\n=== QUESTION TYPE DISTRIBUTION ===")
    qtype_counts = df['question_type'].value_counts()
    for qtype, count in qtype_counts.items():
        print(f"  {qtype}: {count} ({count/len(df)*100:.1f}%)")

    print("\n=== CONTENT RICHNESS ANALYSIS ===")
    content_words = []
    for query in queries:
        content_words.extend(remove_stopwords(query))
    content_word_freq = Counter(content_words)
    print(f"Total unique content words (after stopword removal): {len(content_word_freq)}")
    print("Top content words:")
    for word, count in content_word_freq.most_common(20):
        print(f"  '{word}': {count}")

    print("\n=== QUERY COMPLEXITY METRICS ===")
    content_word_ratio = df['unique_words'].values / df['word_count'].values
    print(f"Avg unique word ratio: {content_word_ratio.mean():.2f}")
    print(f"Queries with high uniqueness (>0.8): {sum(content_word_ratio > 0.8)}/{len(df)} = {sum(content_word_ratio > 0.8)/len(df)*100:.1f}%")

    # Near-duplicate detection (optional, requires embedding model)
    try:
        manager = get_embedding_manager()
        duplicate_pairs = near_duplicate_queries(queries[:100], manager)  # limit to 100 for speed
        if duplicate_pairs:
            print("\n=== NEAR-DUPLICATE QUERIES (top 5) ===")
            for q1, q2, sim in duplicate_pairs[:5]:
                print(f"  Similarity {sim:.3f}: {q1[:50]}... vs {q2[:50]}...")
        else:
            print("\n=== NEAR-DUPLICATE QUERIES: none found ===")
    except Exception as e:
        print(f"Near-duplicate detection skipped: {e}")

    return df

def analyze_course_alignment():
    results_dir = '/home/admin/LLM/LLM/01/web/experiments/results'
    configs = ['bm25_default', 'vector_default', 'hybrid_default']
    confusion = {}
    for config in configs:
        path = f'{results_dir}/{config}.json'
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        results = [r for r in data['results'] if r['k'] == 5]
        # 2x2 matrix: [ [success_same, success_diff], [fail_same, fail_diff] ]
        mat = [[0,0],[0,0]]
        for r in results:
            success = r['success']
            found_course = r.get('found_course', '')
            expected_course = r.get('expected_course', '')  # not stored? Actually we have expected_course? In BenchmarkRunner it's stored as 'expected_course'? Wait, in `all_results` we store 'expected_course'? In benchmark_runner.py we stored 'expected_course'? No, we didn't. Need to get from eval_set.
            # Instead, we need to map query to its course from eval_set.
            # Simpler: we recompute by loading eval_set.
        # This is messy; we'll do a different approach: join with eval_set.
        print(f"Course alignment for {config} not implemented in this version; see cross-analysis below.")
    return

def cross_analysis():
    """Join eval set properties with benchmark results for BM25, Vector, Hybrid."""
    from eval.eval_set import get_eval_set_from_es
    from eval.eda import classify_question_type  # if defined in same file; otherwise move function inside
    import pandas as pd
    import numpy as np
    from collections import Counter

    eval_set = get_eval_set_from_es()
    # Create a dict mapping query -> course and other metadata
    meta = {}
    for item in eval_set:
        doc = item['original_doc']
        query = doc.get('question', '')
        if query:
            meta[query] = {
                'course': doc.get('course', ''),
                'expected_id': item['expected_id'],
                'length': len(query),
                'word_count': len(query.split()),
                'question_type': classify_question_type(query)
            }
    # Load results for each config
    configs = ['bm25_default', 'vector_default', 'hybrid_default']
    results_by_config = {}
    for config in configs:
        results_by_config[config] = load_benchmark_results(config)  # define this function or inline
    # Build dataframe
    rows = []
    for query, m in meta.items():
        row = {
            'query': query,
            'course': m['course'],
            'query_length': m['length'],
            'word_count': m['word_count'],
            'question_type': m['question_type']
        }
        for config in configs:
            res = results_by_config[config].get(query, {})
            row[f'success_{config}'] = res.get('success', False)
            row[f'score_{config}'] = res.get('score', 0.0)
            row[f'found_course_{config}'] = res.get('found_course', '')
        rows.append(row)
    df = pd.DataFrame(rows)

    print("\n=== CROSS-ANALYSIS: PER-QUERY SUCCESS ===")
    for config in configs:
        success_rate = df[f'success_{config}'].mean()
        print(f"{config}: recall@5 = {success_rate:.2%}")

    print("\n=== WIN/LOSS COMPARISON ===")
    bm25_wins = ((df['success_bm25_default'] == True) & (df['success_vector_default'] == False)).sum()
    vector_wins = ((df['success_vector_default'] == True) & (df['success_bm25_default'] == False)).sum()
    both = ((df['success_bm25_default'] == True) & (df['success_vector_default'] == True)).sum()
    neither = ((df['success_bm25_default'] == False) & (df['success_vector_default'] == False)).sum()
    print("BM25 vs Vector:")
    print(f"  BM25 only wins: {bm25_wins}")
    print(f"  Vector only wins: {vector_wins}")
    print(f"  Both: {both}")
    print(f"  Neither: {neither}")

    # Stratify by query length
    print("\n=== STRATIFIED BY QUERY LENGTH ===")
    bins = [0, 10, 30, 60, 100, 200]
    labels = ['0-10', '11-30', '31-60', '61-100', '100+']
    df['len_group'] = pd.cut(df['query_length'], bins=bins, labels=labels)
    for config in configs:
        print(f"\n{config} recall by length group:")
        grouped = df.groupby('len_group')[f'success_{config}'].mean()
        for group, rate in grouped.items():
            print(f"  {group}: {rate:.2%}")

    # Stratify by question type
    print("\n=== STRATIFIED BY QUESTION TYPE ===")
    for qtype in df['question_type'].unique():
        print(f"\nQuestion type: {qtype}")
        subset = df[df['question_type'] == qtype]
        for config in configs:
            rate = subset[f'success_{config}'].mean()
            print(f"  {config}: {rate:.2%}")

    # Per-course recall heatmap (table)
    print("\n=== PER-COURSE RECALL@5 ===")
    courses = df['course'].unique()
    rec_table = []
    for course in courses:
        sub = df[df['course'] == course]
        row = {'course': course}
        for config in configs:
            row[config] = sub[f'success_{config}'].mean()
        rec_table.append(row)
    rec_df = pd.DataFrame(rec_table)
    print(rec_df.round(4))

    # Cross-course contamination: when success is False, where did the top hit come from?
    print("\n=== CROSS-COURSE CONTAMINATION (for failures) ===")
    for config in configs:
        failures = df[df[f'success_{config}'] == False]
        if len(failures) == 0:
            continue
        found_courses = failures[f'found_course_{config}'].value_counts()
        print(f"\n{config}: {len(failures)} failures")
        for found_course, count in found_courses.head(5).items():
            print(f"  Retrieved from course '{found_course}': {count} ({count/len(failures)*100:.1f}%)")

    # Score distribution for hits vs misses
    print("\n=== SCORE DISTRIBUTION (for BM25) ===")
    scores_hit = df[df['success_bm25_default'] == True]['score_bm25_default']
    scores_miss = df[df['success_bm25_default'] == False]['score_bm25_default']
    print(f"Hit scores: mean={scores_hit.mean():.2f}, median={scores_hit.median():.2f}")
    print(f"Miss scores: mean={scores_miss.mean():.2f}, median={scores_miss.median():.2f}")

    # Correlation query length vs success
    print("\n=== CORRELATION (query length vs success) ===")
    for config in configs:
        corr = df['query_length'].corr(df[f'success_{config}'].astype(float))
        print(f"{config}: {corr:.3f}")

    return df

def analyze_error_cases():
    results_dir = '/home/admin/LLM/LLM/01/web/experiments/results'
    configs = ['bm25_default', 'vector_default', 'hybrid_default']
    for config in configs:
        path = f'{results_dir}/{config}.json'
        if not os.path.exists(path):
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        k5_results = [r for r in data['results'] if r['k'] == 5]
        failures = [r for r in k5_results if not r['success']]
        successes = [r for r in k5_results if r['success']]
        print(f"\n=== {config.upper()} ERROR ANALYSIS ===")
        print(f"Total queries: {len(k5_results)}")
        print(f"Failures: {len(failures)} ({len(failures)/len(k5_results)*100:.1f}%)")
        if failures:
            # Average query length of failures vs successes
            # Need to get query length from queries (we can compute on the fly)
            fail_lens = [len(r['query']) for r in failures]
            succ_lens = [len(r['query']) for r in successes]
            print(f"Average query length of failures: {np.mean(fail_lens):.1f} chars")
            print(f"Average query length of successes: {np.mean(succ_lens):.1f} chars")
            # Most common courses among failures
            fail_courses = [r.get('found_course', '') for r in failures]
            course_counts = Counter(fail_courses)
            print("Top courses retrieved in failures:")
            for course, count in course_counts.most_common(5):
                print(f"  {course}: {count}")
            # Sample failures with high edit distance (if we had edit_distance field)
            # For now, just show sample failures
            print("Sample failure queries:")
            for i, f in enumerate(failures[:5]):
                print(f"  {i+1}. {f['query'][:80]}...")
    return

if __name__ == "__main__":
    df_meta = analyze_dataset()
    # Note: cross_analysis requires benchmark results to exist. Run after benchmarks.
    df_cross = cross_analysis()
    #analyze_course_alignment()  # placeholder
    analyze_error_cases()