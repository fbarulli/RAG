
import json
from pathlib import Path
from collections import Counter

path = Path('production_pipeline/p01_data_cleaning/data/processed/eval_queries_tiered.jsonl')

if not path.exists():
    print(f'File not found: {path}')
else:
    tiers = Counter()
    strategies = Counter()
    total = 0
    
    with open(path) as f:
        for line in f:
            if line.strip():
                doc = json.loads(line)
                tiers[doc.get('query_type', 'unknown')] += 1
                # Extract strategy from id: query__{strategy}__...
                strategy = doc['id'].split('__')[1] if '__' in doc['id'] else 'unknown'
                strategies[strategy] += 1
                total += 1
    
    print(f'Total queries: {total}')
    print(f'\nBy tier:')
    for tier, count in tiers.most_common():
        print(f'  {tier}: {count}')
    print(f'\nBy strategy:')
    for strategy, count in strategies.most_common():
        print(f'  {strategy}: {count}')
