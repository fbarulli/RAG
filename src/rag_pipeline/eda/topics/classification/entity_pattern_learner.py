from rag_pipeline.eda.core.paths import Paths
"""
def load_questions(data_path: Path) -> List[str]:

    Ingests the raw evaluation dataset and extracts student/user questions.

    I/O: data_path (Path) -> List[str]



def extract_missed_terms(questions: List[str], nlp: spacy.Language) -> Counter:

    Identifies text collections where the current model failed to find 
    entities, tokenizes them, filters noise, and counts raw term frequencies.

    I/O: questions (List[str]), nlp (spacy.Language) -> Counter



def suggest_patterns(missed_terms: Counter, min_count: int = 3) -> Dict[str, List[str]]:

    Applies heuristic rules to frequently missed terms to categorize them 
    into draft entity classes (TOOL, ERROR, LANGUAGE, ADMIN).

    I/O: missed_terms (Counter), min_count (int) -> Dict[str, List[str]]



def update_entity_ruler(nlp: spacy.Language, suggestions: Dict[str, List[str]]) -> spacy.Language:

    Mutates and updates the SpaCy pipeline by transforming text suggestions 
    into case-insensitive token match patterns.

    I/O: nlp (spacy.Language), suggestions (Dict[str, List[str]]) -> spacy.Language


"""
from pathlib import Path
import json
import re
from collections import Counter
from typing import Set, Dict, List
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
CUSTOM_STOPWORDS = {'the', 'and', 'for', 'with', 'when', 'what', 'use', 'using', 'can', 'why', 'does', 'from', 'are', 'you', 'has', 'should', 'have', 'how', 'not', 'get', 'your', 'this', 'that', 'but', 'they', 'will', 'would', 'could', 'their', 'them', 'then', 'than', 'some', 'into', 'than', 'then', 'just', 'more'}

def load_questions(data_path: Path) -> List[str]:
    """Load questions from topic_assignments_all.json"""
    with open(data_path) as f:
        data = json.load(f)
        assignments = data['results'][Paths.defaults()['production_model']]['assignments']
        return [a['question'] for a in assignments]

def build_base_nlp() -> spacy.Language:
    """Build base NER pipeline with token-based patterns (case-insensitive)"""
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    ruler = nlp.add_pipe('entity_ruler', config={'overwrite_ents': True})
    base_patterns = {'TOOL': ['docker', 'mlflow', 'dlt', 'spark', 'kestra', 'dbt', 'terraform', 'kafka', 'airflow', 'kubernetes', 'gcp', 'aws', 'bigquery', 'github', 'jupyter', 'pandas', 'git', 'postgres', 'redis', 'mongodb', 'wsl', 'vscode', 'vs code', 'codespaces', 'github codespaces', 'xgboost', 'mage', 'pipenv', 'scikit', 'sklearn', 'lambda', 'windows', 'linux', 'google', 'openai', 'grafana', 'waitress', 'pgcli', 'localstack', 'adminer', 'kinesis', 'iam', 'onnx', 'kaggle'], 'LANGUAGE': ['python', 'sql', 'pyspark', 'java', 'bash', 'scala', 'r'], 'CONCEPT': ['embedding', 'embeddings', 'precision', 'recall', 'rmse', 'regression', 'classification', 'overfitting', 'gradient', 'feature', 'features', 'columns', 'model', 'accuracy', 'f1 score', 'auc', 'roc', 'confusion matrix', 'f-score', 'sparse matrix', 'dense matrix'], 'ADMIN': ['certificate', 'homework', 'deadline', 'cohort', 'project', 'course', 'capstone', 'graduation', 'office hours', 'graduate', 'leaderboard', 'prerequisite', 'registration', 'zoomcamp'], 'ERROR': ['error', 'failed', 'cannot', 'unable', 'exception', 'invalid', 'permission denied', 'not found', 'traceback', 'attributeerror', 'importerror', 'valueerror', 'keyerror', 'typeerror', 'connectionerror', 'connectionrefusederror', 'permissionerror', 'timeouterror']}
    patterns = []
    for label, terms in base_patterns.items():
        for term in terms:
            tokens = term.strip().split()
            patterns.append({'label': label, 'pattern': [{'LOWER': t} for t in tokens]})
    ruler.add_patterns(patterns)
    print(f'[build_base_nlp] Added {len(patterns)} token-based patterns (case-insensitive)')
    return nlp

def extract_missed_terms(questions: List[str], nlp: spacy.Language) -> Counter:
    """Extract terms from questions where no entities were detected"""
    missed_terms = Counter()
    for doc in nlp.pipe(questions, batch_size=64):
        if not doc.ents:
            words = re.findall('\\b[a-z][a-z]{2,}\\b', doc.text.lower())
            filtered = [w for w in words if w not in STOP_WORDS and w not in CUSTOM_STOPWORDS]
            missed_terms.update(filtered)
    return missed_terms

def suggest_patterns(missed_terms: Counter, min_count: int=3) -> Dict[str, List[str]]:
    """Suggest entity patterns based on frequent terms in missed questions"""
    suggestions = {'TOOL': [], 'ERROR': [], 'ADMIN': [], 'CONCEPT': [], 'LANGUAGE': []}
    for term, count in missed_terms.most_common(50):
        if count < min_count:
            continue
        term_lower = term.lower()
        if any((x in term_lower for x in ['could', 'recognized', 'insufficient', 'missing', 'stuck', 'timeout', 'crash', 'failed'])):
            suggestions['ERROR'].append(term)
        elif any((x in term_lower for x in ['leaderboard', 'office', 'books', 'resources', 'zoom', 'call', 'certificate', 'deadline'])):
            suggestions['ADMIN'].append(term)
        elif any((x in term_lower for x in ['wsl', 'curl', 'wget', 'chrome', 'vm', 'vms', 'remote', 'html', 'linux', 'codespaces', 'compute', 'engine'])):
            suggestions['TOOL'].append(term)
        elif term_lower in ['bash', 'shell', 'powershell']:
            suggestions['LANGUAGE'].append(term)
    return suggestions

def update_entity_ruler(nlp: spacy.Language, suggestions: Dict[str, List[str]]):
    """Add suggested patterns to the entity ruler (token-based, case-insensitive)"""
    ruler = nlp.get_pipe('entity_ruler')
    patterns = []
    for label, terms in suggestions.items():
        for term in terms:
            if term and len(term) > 2:
                tokens = term.strip().split()
                patterns.append({'label': label, 'pattern': [{'LOWER': t} for t in tokens]})
    ruler.add_patterns(patterns)
    print(f'Added {len(patterns)} new token-based patterns from missed questions')
    return nlp
if __name__ == '__main__':
    data_path = Path('rag_pipeline/p02_eda/experiments/topic_assignments_all.json')
    questions = load_questions(data_path)
    print(f'Loaded {len(questions)} questions')
    nlp = build_base_nlp()
    missed_terms = extract_missed_terms(questions, nlp)
    print(f'Extracted {len(missed_terms)} unique terms from missed questions')
    suggestions = suggest_patterns(missed_terms, min_count=3)
    print('\nSuggested new patterns:')
    for label, terms in suggestions.items():
        if terms:
            print(f'\n  {label}:')
            for term in terms[:10]:
                print(f'    - {term}')
    nlp = update_entity_ruler(nlp, suggestions)
    detected = 0
    misses = []
    for q in questions:
        doc = nlp(q)
        if doc.ents:
            detected += 1
        else:
            misses.append(q)
    print(f"\n{'=' * 60}")
    print(f'COVERAGE: {detected}/{len(questions)} ({detected / len(questions) * 100:.1f}%)')
    print(f'MISSES: {len(misses)}')
    print(f"{'=' * 60}")
    if misses:
        print('\nSample misses (first 10):')
        for i, q in enumerate(misses[:10], 1):
            print(f'{i}. {q[:100]}')