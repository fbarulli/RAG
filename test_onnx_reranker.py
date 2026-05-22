from rag_pipeline.ingestion._onnx_cross_encoder import ONNXCrossEncoder

# Test with a small reranker (fast on CPU)
reranker = ONNXCrossEncoder(
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
    provider="CPUExecutionProvider",
    max_length=512
)

pairs = [
    ("What is the best way to learn Python?", "Python is a great language for beginners."),
    ("How do I install Qdrant?", "You can run Qdrant with Docker."),
    ("What is RAG?", "RAG stands for Retrieval Augmented Generation.")
]

scores = reranker.predict(pairs, batch_size=8, show_progress_bar=True)
print("Scores:", scores)
print("✅ Test passed!")
