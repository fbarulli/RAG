"""
Interactive embedding space visualization with plotly.
Fits UMAP once, saves the reducer, then projects new queries instantly.

Output: analysis/embedding_viz.html   (interactive 3D plot)
        analysis/umap_reducer.joblib  (reusable reducer)
        analysis/doc_embeddings.npy   (cached embeddings)
        analysis/doc_meta.json        (cached metadata)

Run:    uv run python analysis/embedding_viz.py
Re-fit: uv run python analysis/embedding_viz.py --refit
Query:  uv run python analysis/embedding_viz.py --query "your question here"
"""
import sys, os, json, joblib, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import plotly.graph_objects as go
from sentence_transformers import SentenceTransformer
from umap import UMAP

# ── Config ───────────────────────────────────────────────────────────────────
MODEL        = 'BAAI/bge-base-en-v1.5'
REDUCER_FILE = 'analysis/umap_reducer.joblib'
EMBED_FILE   = 'analysis/doc_embeddings.npy'
META_FILE    = 'analysis/doc_meta.json'
REDUCED_FILE = 'analysis/doc_reduced.npy'
OUTPUT       = 'analysis/embedding_viz.html'

COURSE_COLORS = {
    'de-zoomcamp':    '#4e9af1',
    'llm-zoomcamp':   '#f4a22d',
    'ml-zoomcamp':    '#3ecf8e',
    'mlops-zoomcamp': '#e05c6b',
}

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--refit', action='store_true', help='Force re-fit UMAP')
parser.add_argument('--query', type=str, default=None, help='Project a new query into the space')
args = parser.parse_args()

# ── Load embedding model (always needed) ─────────────────────────────────────
print(f"Loading embedding model: {MODEL}")
embed_model = SentenceTransformer(MODEL)

# ── Fit or load UMAP ──────────────────────────────────────────────────────────
cache_exists = all(os.path.exists(f) for f in [REDUCER_FILE, EMBED_FILE, META_FILE, REDUCED_FILE])

if cache_exists and not args.refit:
    print("Loading cached embeddings and UMAP reducer...")
    reducer      = joblib.load(REDUCER_FILE)
    doc_embeddings = np.load(EMBED_FILE)
    doc_reduced  = np.load(REDUCED_FILE)
    with open(META_FILE) as f:
        doc_meta = json.load(f)
    print(f"  {len(doc_meta)} documents loaded from cache.")
else:
    print("Fitting UMAP (one-time, ~2 min)...")
    with open('data_cleaning/data/processed/clean.jsonl') as f:
        docs = [json.loads(line) for line in f]

    doc_texts = [d['question'] for d in docs]
    print(f"  Embedding {len(doc_texts)} documents...")
    doc_embeddings = embed_model.encode(doc_texts, show_progress_bar=True, batch_size=64)

    print("  Reducing with UMAP (3D)...")
    reducer = UMAP(n_components=3, random_state=42, n_jobs=1, n_neighbors=15, min_dist=0.1)
    doc_reduced = reducer.fit_transform(doc_embeddings)

    doc_meta = [
        {
            'question': d['question'],
            'course':   d['course'],
            'section':  d.get('section', ''),
        }
        for d in docs
    ]

    os.makedirs('analysis', exist_ok=True)
    np.save(EMBED_FILE, doc_embeddings)
    np.save(REDUCED_FILE, doc_reduced)
    joblib.dump(reducer, REDUCER_FILE)
    with open(META_FILE, 'w') as f:
        json.dump(doc_meta, f)
    print(f"  Saved reducer → {REDUCER_FILE}")

# ── Query mode: project and report, then exit ────────────────────────────────
if args.query:
    q = args.query
    print(f"\nProjecting query: '{q}'")
    q_emb = embed_model.encode([q])
    q_xy  = reducer.transform(q_emb)[0]

    # Find 5 nearest neighbours by cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sims   = cosine_similarity(q_emb, doc_embeddings)[0]
    top5   = np.argsort(sims)[::-1][:5]

    print(f"\nNearest documents:")
    for rank, idx in enumerate(top5, 1):
        m = doc_meta[idx]
        print(f"  {rank}. [{m['course']}] {m['question'][:80]}  (sim={sims[idx]:.3f})")

    print(f"\nQuery 3D coords: ({q_xy[0]:.3f}, {q_xy[1]:.3f}, {q_xy[2]:.3f})")
    print("Re-run without --query to add it to the HTML plot.")
    sys.exit(0)

# ── Build plot ────────────────────────────────────────────────────────────────
print("Building plot...")
courses  = np.array([m['course']   for m in doc_meta])
sections = np.array([m['section']  for m in doc_meta])
questions= np.array([m['question'] for m in doc_meta])

fig = go.Figure()

for course, color in COURSE_COLORS.items():
    mask = courses == course
    if not mask.any():
        continue
    hover = [
        f"<b>{q[:80]}</b><br><i>{s}</i>"
        for q, s in zip(questions[mask], sections[mask])
    ]
    fig.add_trace(go.Scatter3d(
        x=doc_reduced[mask, 0],
        y=doc_reduced[mask, 1],
        z=doc_reduced[mask, 2],
        mode='markers',
        marker=dict(size=3, color=color, opacity=0.6),
        name=course,
        text=hover,
        hovertemplate='%{text}<extra></extra>',
    ))

fig.update_layout(
    title=dict(text='FAQ Embedding Space — 3D by Course', font=dict(size=18)),
    width=1100,
    height=750,
    hovermode='closest',
    legend=dict(title='Course', itemsizing='constant'),
    scene=dict(
        xaxis=dict(title='UMAP 1', showbackground=False, showgrid=True),
        yaxis=dict(title='UMAP 2', showbackground=False, showgrid=True),
        zaxis=dict(title='UMAP 3', showbackground=False, showgrid=True),
        bgcolor='rgb(10,10,20)',
    ),
    paper_bgcolor='rgb(15,15,25)',
    font=dict(color='white'),
)

fig.write_html(OUTPUT, include_plotlyjs='cdn')
print(f"\nSaved: {OUTPUT}")
print(f"  {len(doc_meta)} documents across {len(COURSE_COLORS)} courses")
print(f"\nTo project a query:")
print(f"  uv run python analysis/embedding_viz.py --query 'how do I submit homework?'")
print(f"  uv run python analysis/embedding_viz.py --refit   # force re-fit UMAP")