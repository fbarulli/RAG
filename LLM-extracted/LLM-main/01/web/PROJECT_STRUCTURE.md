# RAG FAQ Search for DataTalks.Club Zoomcamps

Production RAG system serving 4 courses (ML, DE, MLOps, LLM) with **92.4% R@5** in Slack channels.

## Project Structure

```text
llm-zoomcamp-faq/
│
├── data/
│   ├── raw/                          # Downloaded markdown from GitHub
│   ├── processed/                    # clean.jsonl, parsed.jsonl
│   ├── splits/                       # train.jsonl (80%), test.jsonl (20%)
│   └── metadata/                     # images.json (image references)
│
├── ingest/
│   ├── download.py                   # Pull FAQ markdown from DataTalksClub/faq
│   ├── parse.py                      # Extract id, question, answer, course, section
│   ├── dedup.py                      # 95% similarity threshold deduplication
│   ├── split.py                      # 80/20 train/test split by course
│   ├── elasticsearch.py              # Index to ES faqs_complete with question_vector
│   ├── qdrant.py                     # Index to Qdrant faqs collection
│   └── run_all.sh                    # Complete pipeline runner
│
├── retrieval/
│   ├── __init__.py
│   ├── bm25.py                       # Lexical search with question/text boosting
│   ├── vector.py                     # Cosine similarity on all-MiniLM-L6-v2
│   ├── hybrid.py                     # RRF or linear weighted fusion
│   ├── qdrant.py                     # Qdrant vector search
│   └── reranker.py                   # Cross-encoder reranking (ms-marco-MiniLM-L-6-v2)
│
├── evaluation/
│   ├── __init__.py
│   ├── eval_set.py                   # Core evaluation dataset (390 queries)
│   │
│   ├── queries/
│   │   ├── generate.py               # Generate test queries with LLM
│   │   ├── prompts.json              # 3 strategies: grounded_analyst, creative_student, chaos_monkey
│   │   └── prompts_cag.json          # CAG-specific prompts
│   │
│   ├── judges/
│   │   ├── context_sufficiency.py    # LLM-as-judge for retrieval quality
│   │   ├── factual_correctness.py    # RAGAS-style factual correctness (8B model)
│   │   ├── calibrate.py              # Judge calibration against human labels
│   │   ├── classify_failures.py      # Failure mode classification
│   │   └── shared.py                 # llm_call, rate limiting, retries
│   │
│   ├── benchmarks/
│   │   ├── run.py                    # Full benchmark runner (11 configs)
│   │   ├── compare_models.py         # Embedding model comparison (BGE, E5)
│   │   ├── compare_retrievers.py     # BM25 vs vector vs hybrid vs Qdrant
│   │   ├── test_variations.py        # All config variations with generated queries
│   │   ├── test_hyde.py              # HyDE query expansion
│   │   ├── test_no_filter.py         # Open search vs course filter
│   │   └── holdout.py                # Holdout evaluation on unseen documents
│   │
│   ├── analysis/
│   │   ├── dashboard.py              # Interactive dashboard with plots
│   │   ├── ab_test.py                # A/B significance testing
│   │   ├── significance_test.py      # Statistical significance (p-values)
│   │   ├── visualizer.py             # Generate plot PNGs
│   │   └── failure_analysis.py       # Deep dive on failed queries
│   │
│   ├── ragas/
│   │   ├── evaluate.py               # RAGAS evaluation (FactualCorrectness)
│   │   ├── cache.py                  # Result caching to avoid re-runs
│   │   └── prompts.py                # RAGAS prompts
│   │
│   └── cag/
│       ├── generate.py               # Generate cached answers for all FAQs
│       ├── evaluate.py               # Evaluate CAG quality vs original FAQ
│       └── store.py                  # CAG answer storage
│
├── src/
│   ├── clients/
│   │   ├── elasticsearch.py          # ES connection pool
│   │   └── qdrant.py                 # Qdrant client wrapper
│   ├── embedding/
│   │   └── service.py                # Embedding service (BGE, E5, MiniLM)
│   ├── core/
│   │   ├── cache.py                  # LLM response cache
│   │   ├── config.py                 # Configuration manager
│   │   ├── rate_limiter.py           # Token-based rate limiting
│   │   └── logger.py                 # Structured logging
│   ├── llm/
│   │   ├── models.py                 # LLM wrappers (NVIDIA 70B, 8B)
│   │   ├── prompt_manager.py         # Prompt template management
│   │   └── langfuse.py               # Langfuse tracing & observability
│   └── guardrails.py                 # Output validation
│
├── experiments/
│   ├── results/                      # Benchmark output JSONs
│   │   ├── bm25_default.json
│   │   ├── bm25_balanced.json
│   │   ├── hybrid_70_30_vec.json
│   │   ├── qdrant_default.json
│   │   └── variations_*.json
│   ├── judge/                        # Judge progress CSVs
│   ├── plots/                        # Saved visualization PNGs
│   ├── queries/
│   │   ├── eval_queries.json         # 390 generated test queries
│   │   └── topic0_queries.json       # Topic 0 sub-cluster queries
│   ├── cag_answers.json              # 152 CAG answers (of 1140)
│   ├── ragas_scores.json             # RAGAS evaluation results
│   └── llm_cache.json                # LLM response cache
│
├── configs/
│   ├── elasticsearch.json            # ES index settings & mappings
│   ├── qdrant.json                   # Qdrant collection config
│   ├── bm25.json                     # BM25 boosting weights
│   └── settings.json                 # API keys, model names, batch sizes
│
├── notebooks/
│   ├── eda.ipynb                     # EDA with BERTopic clustering
│   ├── eval_dashboard.ipynb          # Interactive evaluation dashboard
│   └── topic_modeling.ipynb          # Topic analysis (20 clusters)
│
├── docs/
│   ├── README.md                     # Main documentation
│   ├── ARCHITECTURE.md               # Pipeline architecture diagram
│   ├── RESULTS.md                    # Key findings (86.9% open, 92.4% filtered)
│   ├── DEPLOYMENT.md                 # Slack integration, course filtering
│   └── TROUBLESHOOTING.md            # Common issues & solutions
│
├── scripts/
│   ├── run_benchmark.sh              # Run full benchmark suite
│   ├── run_judge.sh                  # Run LLM-as-judge evaluation
│   ├── run_ragas.sh                  # Run RAGAS evaluation
│   └── clean_results.sh              # Clean old experiment artifacts
│
├── docker-compose.yaml               # ES + Qdrant + app containers
├── docker-compose-qdrant.yaml        # Standalone Qdrant
├── pyproject.toml                    # Python dependencies
├── Makefile                          # Common commands (ingest, benchmark, judge)
└── .env.example                      # Environment variable template