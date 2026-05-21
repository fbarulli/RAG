Based on what I've seen, here are the most impactful directions ranked by likely ROI:
High impact, lower effort

Query expansion — before retrieval, expand the query with synonyms or related terms. Your entity boosting is already doing a light version of this with NER, but you could add LLM-generated query variants and take the best hit across all of them (you already have the LLM client)
Reciprocal Rank Fusion over rerankers — instead of picking one reranker, combine all their rankings with RRF. Your disagreement analysis will show if different rerankers are good at different query types — combining them could beat any single one
Score calibration — TinyBERT and MiniLM are outputting scores of 0.88-0.90 on failures, which suggests they're not well-calibrated for your domain. A simple sigmoid or min-max rescaling per reranker could help

Medium impact, medium effort

Fine-tune the best reranker — take your test set, generate hard negatives (the wrong answers the rerankers are currently ranking above the correct one), and fine-tune bge-reranker-base on your domain. This is probably the single highest ceiling move
Chunk-level answer enrichment — your answers are already clean but adding the question text to the indexed document (question + answer concatenated) before embedding could improve bi-encoder recall, which is the ceiling for reranker performance
Two-stage retrieval — retrieve 100 candidates instead of 40 with the bi-encoder, then rerank. Your current setup might be losing the correct doc before reranking even starts on harder queries

Lower effort, diagnostic value

Per-topic reranker routing — your topic map is already built. Some topics might respond better to different rerankers — route by topic based on the failure analysis results
OOD query handling — your test set has OOD queries flagged. Separating their metrics would give a cleaner picture of in-distribution performance









Yes, exactly. Right now we're just running pre-trained ONNX rerankers against your test set with no domain adaptation at all. The correct pipeline should be:

Train set → fine-tune the reranker on your domain (query, positive answer, hard negatives)
Holdout/test set → evaluate the fine-tuned reranker

What we're currently doing is skipping step 1 entirely and just benchmarking off-the-shelf models on your test set. That's why the scores are poor — these rerankers have never seen your FAQ domain.