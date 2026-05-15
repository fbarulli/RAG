# /home/admin/LLM/LLM/01/web/src/cache.py

import hashlib
import json
import numpy as np
from redis import Redis
from sentence_transformers import SentenceTransformer

def cosine_similarity_vec(a, b):
    """Calculate cosine similarity between two 1D vectors."""
    a = np.array(a).reshape(1, -1)
    b = np.array(b).reshape(1, -1)
    return np.dot(a, b.T)[0][0] / (np.linalg.norm(a) * np.linalg.norm(b))

class SemanticCache:
    def __init__(self, redis_host="localhost", similarity_threshold=0.95):
        self.redis = Redis(host=redis_host, decode_responses=True)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.threshold = similarity_threshold
    
    def _get_embedding(self, text):
        return self.model.encode(text).tolist()
    
    def get(self, query):
        query_embed = self._get_embedding(query)
        # Find similar cached queries
        for cached_query in self.redis.scan_iter("cache:*"):
            cached_embed = json.loads(self.redis.hget(cached_query, "embedding"))
            similarity = cosine_similarity_vec(query_embed, cached_embed)
            if similarity > self.threshold:
                return self.redis.hget(cached_query, "response")
        return None
    
    def set(self, query, response):
        key = f"cache:{hashlib.md5(query.encode()).hexdigest()[:16]}"
        self.redis.hset(key, mapping={
            "query": query,
            "response": response,
            "embedding": json.dumps(self._get_embedding(query))
        })
        self.redis.expire(key, 86400)  # 24 hour TTL