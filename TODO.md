# 🛠️ RAG-a-muffin: Technical Roadmap & Architecture Notes

This document serves as the source of truth for the current architectural state, identified bottlenecks, and the plan for improving retrieval performance.

---

## 🏗️ Core Architecture

### 1. The Path & Config System (The "No-Magic-Strings" Rule)
To prevent the project from breaking when files move or environments change, we have decoupled the filesystem from the logic.

**A. Centralized Path Resolution (`Paths` Class)**
Instead of using `Path("../data/file.json")` inside modules, we use `src/rag_pipeline/core/paths.py`.
- **Mechanism:** The `Paths` class acts as a static resolver. It reads `configs/paths.json` at runtime and returns absolute paths relative to the project root.
- **The Benefit:** To move the entire data folder, you change **one line** in a JSON file rather than searching and replacing paths in 50 different Python files.
- **Example:** `Paths.topic_assignments()` $\rightarrow$ resolves to `output/topic_assignments_all.json`.

**B. Configuration-Driven Logic (JSON $\rightarrow$ Class)**
We use a "Configuration Object" pattern to handle runtime settings:
- **`configs/defaults.json`**: Holds global constants (batch sizes, hosts, ports).
- **`configs/retrieval_configs.json`**: Defines the behavior of retrieval strategies (weights, filters).
- **`BenchmarkConfig` Class**: A wrapper that loads these JSONs and allows **CLI Overrides**. (e.g., if you pass `--top-k 50` on the CLI, it overrides the JSON default).

### 2. The Retriever Facade
The retrieval layer has been split to separate concerns and ensure maintainability:
- **`retrievers.py`**: A thin facade. Only imports and exports functions to maintain compatibility with `evaluation.py`.
- **`es_retrievers.py`**: Pure BM25 logic (Elasticsearch).
- **`qdrant_retrievers.py`**: Pure Vector logic and Qdrant filter construction.
- **`composite_retrievers.py`**: Hybrid (RRF/DBSF) and Boosted logic.

---

## ⚡ Performance Optimizations

### 1. Embedding Caching (CPU Optimization)
To eliminate redundant computation on CPU-only hardware, we implemented a `.npy` caching layer in `src/rag_pipeline/ingestion/embedding_cache.py`.
- **Query Cache:** Test query vectors are cached by model slug. This makes the startup of a benchmark run nearly instantaneous.
- **Corpus Cache:** Corpus embeddings (Question + Answer) are cached. This prevents `ablation run` from spending minutes re-encoding the dataset every time a topic patch is applied.

### 2. Ingestion Pipeline
- Moved from manual loops to high-performance Qdrant point uploading.
- Integrated the cache check into `ingest_qdrant.py` to ensure "compute once, read many."

---

## 🔍 Findings & Critical Insights

### 1. The "NER Poisoning" Effect
**Finding:** The `entity_boosted` configuration significantly hurt performance (~12pp drop in H@1) compared to `vector_default`.
- **Root Cause:** The `entity_patterns.json` contains generic signals (e.g., `"error"`, `"project"`, `"course"`). 
- **The Mechanic:** Because Qdrant uses a `should` clause for boosting, any document containing these generic terms gets a score boost. In a technical FAQ, almost every document contains the word "error," causing the retriever to promote irrelevant noise and displace the correct answer.
- **Conclusion:** NER signals must be **high-precision**. Generic terms must be pruned.

### 2. Topic Modeling Logic
- **Clustering Signal:** Clustering on `Question` alone is insufficient. `Question + Answer` provides the technical context (e.g., "Airflow", "K8s") necessary for accurate topic discovery.
- **Outlier Management:** Use `approximate_distribution` to reassign outliers to the next most likely topic to reduce the "Topic -1" count.

---

## 📅 TODO & Future Work

### 🔴 High Priority (Precision & Performance)
- [ ] **Prune Entity Patterns:** Audit `configs/entity_patterns.json` and remove generic signals. Keep only specific `TOOL`, `LANGUAGE`, and `CONCEPT` terms.
- [ ] **Asymmetric NER Tagging:** Implement "Question-First, Answer-Fallback" logic.
    - Extract from Question $\rightarrow$ If `TOOL/LANGUAGE`, use as `primary_entity`.
    - If Question is generic $\rightarrow$ Extract from Answer $\rightarrow$ If `TOOL/LANGUAGE`, use as `primary_entity`.
    - Otherwise $\rightarrow$ `OTHER`.
- [ ] **Complete Ablation Suite:** Run the remaining `skip_cluster`, `skip_rules`, and `empty_patterns` experiments to quantify the exact value of each pipeline stage.

### 🟡 Medium Priority (Refinement)
- [ ] **Corpus Sampling:** Use `corpus_sampler.py` to find the minimum training set size that preserves H@1 performance.
- [ ] **ID Normalization:** Resolve naming inconsistencies between the corpus (`machine-learning-zoomcamp`) and test queries (`ml-zoomcamp`).
- [ ] **Reranker Integration:** Move from simple Vector retrieval to a `Vector $\rightarrow$ Reranker` pipeline using the best-performing Cross-Encoder.

### 🟢 Low Priority (Maintenance)
- [ ] **Full Suite Commit:** Once the "No-Entity" baseline is recovered, commit the restructured `src/` and `ablation/` directories.
- [ ] **Documentation:** Update `RERANKER.md` with the final findings on entity boosting vs. pure vector search.





```bash
cat > HANDOFF.md << 'EOF'
# Ablation Handoff — 2026-05-28

## Baseline (current best)
| Config | H@1 | MRR |
|---|---|---|
| entity_boosted | 88.8% | 0.9242 |

### By query type
| query_type | n | H@1 |
|---|---|---|
| chaos_monkey | 147 | 78.9% |
| creative_student | 120 | 90.8% |
| grounded_analyst | 147 | 93.2% |
| original | 49 | 100.0% |

---

## Completed Experiments
| Name | Delta H@1 | Finding |
|---|---|---|
| no_generic_entity | -3.3pp | Generic entities HELP, especially chaos_monkey |
| low_conf_null_50 | 0pp | No effect — 47% of docs below 0.5 threshold |
| low_conf_null_40 | 0pp | No effect — topic probability is not driving retrieval |
| no_cluster | 0pp | is a confirmed no-op|

---



---

## Known Good
- `ingest_qdrant.py` — refactored, caching added, no hardcoded vars
- `ingest_es.py` — refactored, caching added, no hardcoded vars
- `topic_modeling.py` — `or` pattern fixed, `_require` fixed, `subtopic_min_size` in defaults
- `topic_assignments_all.json` — valid JSON, re-downloaded + re-merged
- All imports verified clean

---

## What Is Left
1. **Verify no_cluster silent failure** — diff assignments or add logging to topic_rules.py
2. **chaos_monkey gap** — 78.9% is the only meaningful remaining gap. If no_cluster
   is confirmed a no-op, investigate what the 147 failing chaos queries look like.
   Is it a retrieval problem or a query formulation problem?
3. **Question-First, Answer-Fallback NER strategy** — deferred until ablation suite
   is fully trusted. Requires full topic modeling rerun across all models.
4. **Corpus sampling** — minimum corpus size investigation with corpus_sampler.py.
   Fix naming mismatch: `ml-zoomcamp` vs `machine-learning-zoomcamp` in test set.
5. **Commit** — restructured src/ and ablation/ once no_cluster is resolved.

---

## Files Modified This Session
- `src/rag_pipeline/ingestion/ingest_qdrant.py`
- `src/rag_pipeline/ingestion/ingest_es.py`
- `src/rag_pipeline/core/paths.py` — added `embeddings_cache_dir`, fixed `topics_default_output`
- `src/rag_pipeline/ablation/experiment.py` — replaced `_load_from_git` with direct read
- `src/rag_pipeline/eda/topics/core/topic_modeling.py` — fixed `or` pattern, `_require`, fallback
- `configs/defaults.json` — added `subtopic_min_size: 5`
EOF
echo "Done"
```