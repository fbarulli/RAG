from rag_pipeline.eda.core.paths import Paths
"""
Public Functions for Config-Driven NER Pipeline:

def build_ner_from_config(config_path: Path) -> spacy.Language:
    Builds a SpaCy NER pipeline using patterns from a JSON config file.
    I/O: config_path (Path) -> spacy.Language

def tag_questions(questions: List[str], nlp: spacy.Language) -> List[Dict]:
    Tags questions with entities using the configured NER pipeline.
    I/O: questions (List[str]), nlp (spacy.Language) -> List[Dict]


NER pipeline that loads entity patterns from a JSON configuration file.
This makes it easy to add/remove patterns without changing code.
"""
import spacy
from pathlib import Path
import json
from typing import Dict, List, Optional

def load_entity_patterns(config_path: Path) -> Dict[str, List[str]]:
    """Load entity patterns from JSON configuration."""
    with open(config_path) as f:
        return json.load(f)

def build_ner_from_config(config_path: Path) -> spacy.Language:
    """Build a SpaCy NER pipeline using patterns from a JSON config file."""
    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    ruler = nlp.add_pipe('entity_ruler', config={'overwrite_ents': True, 'phrase_matcher_attr': 'LOWER'})
    patterns = load_entity_patterns(config_path)
    all_patterns = []
    for label, term_list in patterns.items():
        for term in term_list:
            all_patterns.append({'label': label, 'pattern': term.lower()})
            if ' ' in term:
                all_patterns.append({'label': label, 'pattern': term.lower().replace(' ', '')})
            if ' ' not in term and term.islower():
                all_patterns.append({'label': label, 'pattern': term.title()})
    ruler.add_patterns(all_patterns)
    print(f'[NER] Loaded {len(all_patterns)} patterns from config')
    return nlp

def tag_questions(questions: List[str], nlp: spacy.Language) -> List[Dict]:
    """Tag questions with entities using the configured NER pipeline."""
    results = []
    for doc in nlp.pipe([q.lower() for q in questions], batch_size=64):
        entities = [{'text': ent.text, 'label': ent.label_} for ent in doc.ents]
        category = 'OTHER'
        primary_entity = None
        for entity in entities:
            if entity['label'] == 'ADMIN':
                category = 'ADMIN'
                primary_entity = entity['text']
                break
            elif entity['label'] == 'ERROR':
                category = 'ERROR'
                primary_entity = entity['text']
                break
            elif entity['label'] in ['TOOL', 'LANGUAGE']:
                category = entity['label']
                primary_entity = entity['text']
                break
            elif entity['label'] == 'CONCEPT' and category == 'OTHER':
                category = 'CONCEPT'
                primary_entity = entity['text']
        results.append({'question': doc.text, 'entities': entities, 'category': category, 'primary_entity': primary_entity})
    return results
if __name__ == '__main__':
    config_path = Path('rag_pipeline/p02_eda/entity_patterns.json')
    nlp = build_ner_from_config(config_path)
    test_questions = ['How to run Python as a startup script?', "HPA instance doesn't run properly", 'Deploying to Digital Ocean', 'What does pandas.DataFrame.info() do?', 'Any advice for adding experience to your LinkedIn profile?', 'What is the difference between OneHotEncoder and DictVectorizer?']
    print('\nTesting NER from config:')
    for q in test_questions:
        doc = nlp(q.lower())
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        print(f'\n{q}')
        print(f'  Entities: {entities}')
    data_path = Path('rag_pipeline/p02_eda/experiments/topic_assignments_all.json')
    if data_path.exists():
        with open(data_path) as f:
            data = json.load(f)
            assignments = data['results'][Paths.defaults()['production_model']]['assignments']
            questions = [a['question'] for a in assignments]
        detected = 0
        for doc in nlp.pipe([q.lower() for q in questions], batch_size=64):
            if doc.ents:
                detected += 1
        print(f"\n{'=' * 60}")
        print(f'Coverage on {len(questions)} questions: {detected}/{len(questions)} ({detected / len(questions) * 100:.1f}%)')
        print(f"{'=' * 60}")