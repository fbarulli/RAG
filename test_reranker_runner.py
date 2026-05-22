from rag_pipeline.ingestion._reranker_runner import RerankerRunner

runner = RerankerRunner(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Default batch size:", runner.onnx_reranker.default_batch_size)

query = "What is the best way to learn Python?"
docs = [
    "Python is a great language for beginners with simple syntax.",
    "You can install Python from the official website.",
    "RAG stands for Retrieval Augmented Generation.",
    "Machine learning models can be fine-tuned on custom data."
]

results = runner.rerank(query, docs, show_progress=False)

print("\nReranking results:")
for doc, score in results[:3]:
    print(f"{score:.4f} -> {doc[:70]}...")
