
import json
from collections import Counter
from pathlib import Path

path = Path("production_pipeline/p02_eda/experiments/topic_assignments.json")
data = json.load(open(path))

assignments = data["assignments"]
metadata = data["metadata"]

print(f"📊 Topic Assignment Summary")
print(f"   Total documents: {metadata['total_documents']}")
print(f"   Topics discovered: {metadata['num_topics']}")
print(f"   Outliers (topic -1): {metadata['outlier_count']} ({metadata['outlier_ratio']:.1%})")
print()

# Topic size distribution
topic_counts = Counter(a["topic"] for a in assignments if a["topic"] != -1)
sizes = sorted(topic_counts.values(), reverse=True)

print(f"📈 Topic Size Distribution (top 15):")
print(f"   {'Rank':<5} {'Topic ID':<10} {'Docs':>6} {'% of Corpus':>12}")
print(f"   {'-'*35}")
for rank, (topic_id, count) in enumerate(topic_counts.most_common(15), 1):
    pct = count / len(assignments) * 100
    print(f"   {rank:<5} {topic_id:<10} {count:>6} {pct:>11.1f}%")

print()
print(f"📉 Size Stats:")
print(f"   Mean: {sum(sizes)/len(sizes):.1f} docs/topic")
print(f"   Median: {sorted(sizes)[len(sizes)//2]} docs/topic")
print(f"   Min: {min(sizes)} | Max: {max(sizes)}")
print(f"   Std Dev: {(sum((x - sum(sizes)/len(sizes))**2 for x in sizes)/len(sizes))**0.5:.1f}")

# Sample topic keywords for top 5 topics (fixed tuple handling)
print()
print(f"🔑 Sample Keywords (top 5 topics):")
for topic_id, _ in topic_counts.most_common(5):
    sample = [a for a in assignments if a["topic"] == topic_id][0]
    keywords = sample.get("keywords", [])
    
    # Handle BERTopic format: [(word, weight), ...] OR flat list of strings
    if keywords and isinstance(keywords[0], (list, tuple)):
        kw_list = [item[0] for item in keywords if isinstance(item, (list, tuple)) and len(item) >= 1]
    else:
        kw_list = [k for k in keywords if isinstance(k, str)]
    
    kw_str = ", ".join(kw_list[:8])
    print(f"   Topic {topic_id}: [{kw_str}]")
