import traceback
import time
from typing import List, Dict, Any, Optional
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from src.logger_config import logger, time_logger
from src.cache import SemanticCache
from src.guardrails import guardrail_filter


from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.schema import Document
from llama_index.core.evaluation import FaithfulnessEvaluator, RelevancyEvaluator
from llama_index.llms.litellm import LiteLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.base.response.schema import Response


from dotenv import load_dotenv
load_dotenv('/home/admin/LLM/LLM/01/web/configs/.env')



import time
from functools import wraps

def rate_limit(calls_per_minute=40):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            last_called[0] = time.time()
            return func(*args, **kwargs)
        return wrapper
    return decorator




class CourseRAGManager:
    def __init__(self, settings: Dict[str, Any]):
        self.settings = settings
        self.es_client: Optional[Elasticsearch] = None
        self.index_name = self.settings.get("index_name", "course-questions")
        self.embedding_model = None
        self.cache = None
        self.llama_index = None
        self.query_engine = None
        self.evaluator_faithfulness = None
        self.evaluator_relevancy = None
        
        if self.settings.get("use_vector", False):
            model_name = self.settings.get("embedding_model", "all-MiniLM-L6-v2")
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_model.encode("warmup")
            logger.info(f"Loaded and warmed up embedding model: {model_name}")
            Settings.embed_model = HuggingFaceEmbedding(model_name=model_name)
        
        if self.settings.get("use_llama_llm", False):
            llm_model = self.settings.get("llama_llm_model", "nvidia_nim/meta/llama-3.1-8b-instruct")
            Settings.llm = LiteLLM(model=llm_model)
            logger.info(f"Loaded LlamaIndex LLM: {llm_model}")
        
        if self.settings.get("use_evaluators", False):
            self._init_evaluators()
        
        if self.settings.get("use_cache", False):
            self.cache = SemanticCache(
                redis_host=self.settings.get("redis_host", "localhost"),
                similarity_threshold=self.settings.get("cache_threshold", 0.95)
            )
            logger.info(f"Cache initialized")
        
        if self.settings.get("build_llama_index", False):
            self._build_llama_index()
    
    def _init_evaluators(self):
        import os
        llm_model = self.settings.get("llama_llm_model", "nvidia_nim/meta/llama-3.1-8b-instruct")
        try:
            llm = LiteLLM(model=llm_model)
            self.evaluator_faithfulness = FaithfulnessEvaluator(llm=llm)
            self.evaluator_relevancy = RelevancyEvaluator(llm=llm)
            logger.info(f"Initialized evaluators with LLM: {llm_model}")
        except Exception as e:
            logger.error(f"Failed to initialize evaluators: {e}")
    
    def _build_llama_index(self):
        import json
        logger.info("Building LlamaIndex from documents...")
        with open("documents.json", "r") as f:
            data = json.load(f)
        
        documents = []
        for course in data:
            course_name = course["course"]
            for doc in course["documents"]:
                question = doc["question"]
                if " - " in question:
                    question = question.split(" - ", 1)[1].strip()
                text = f"Question: {question}\nAnswer: {doc['text']}"
                documents.append(Document(
                    text=text,
                    metadata={
                        "course": course_name,
                        "original_question": question
                    }
                ))
        
        self.llama_index = VectorStoreIndex.from_documents(documents)
        self.query_engine = self.llama_index.as_query_engine()
        logger.info(f"Built LlamaIndex with {len(documents)} documents")
        
    def connect_elasticsearch(self) -> None:
        host = self.settings.get("es_host", "http://localhost:9200")
        try:
            self.es_client = Elasticsearch(host)
            if not self.es_client.ping():
                raise ConnectionError("ES Ping failed")
        except Exception:
            logger.error(f"Connection failed: {traceback.format_exc()}")
            raise

    def _get_query_embedding(self, query: str) -> List[float]:
        if not self.embedding_model:
            return []
        for attempt in range(3):
            try:
                return self.embedding_model.encode(query).tolist()
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt+1} failed: {e}")
                if attempt == 2:
                    return []
                time.sleep(0.1)
        return []

    @time_logger
    def search_faq(self, query: str, override_size: int, course_context: Optional[str] = None) -> List[Dict]:
        if not self.es_client:
            return []

        allowed, guardrail_response = guardrail_filter(query)
        if not allowed:
            logger.info(f"Query blocked by guardrails: {query[:50]}...")
            return [{
                "_id": "blocked",
                "_score": 0.0,
                "_source": {
                    "question": "Query Blocked",
                    "text": guardrail_response,
                    "course": course_context or "guardrail"
                }
            }]

        if self.cache:
            cached_response = self.cache.get(query)
            if cached_response:
                return [{
                    "_id": "cached",
                    "_score": 1.0,
                    "_source": {
                        "question": "Cached Response",
                        "text": cached_response,
                        "course": course_context or "cached"
                    }
                }]

        if self.settings.get("use_llama_query", False) and self.query_engine:
            response = self.query_engine.query(query)
            return [{
                "_id": "llama_response",
                "_score": 1.0,
                "_source": {
                    "question": query,
                    "text": str(response),
                    "course": course_context or "llama"
                }
            }]

        search_type = self.settings.get("search_type", "bm25")
        if search_type == "vector":
            results = self._vector_search(query, override_size, course_context)
        elif search_type == "hybrid":
            results = self._hybrid_search(query, override_size, course_context)
        else:
            results = self._bm25_search(query, override_size, course_context)
        
        if self.cache and results:
            top_response = results[0].get('_source', {}).get('text', '')
            if top_response:
                self.cache.set(query, top_response)
        
        return results
    
    @rate_limit(calls_per_minute=40)
    def evaluate_response(self, query: str, response_text: str, contexts: List[str]) -> Dict[str, Any]:
        """Use LlamaIndex evaluators to assess answer quality."""
        if not self.evaluator_faithfulness or not self.evaluator_relevancy:
            logger.warning("Evaluators not initialized")
            return {"error": "Evaluators not enabled"}
        
        # The LlamaIndex evaluators expect a plain string for the query
        # and the response text directly, not a Response object.
        try:
            faithfulness_result = self.evaluator_faithfulness.evaluate(
                query=query,
                response=response_text,
                contexts=contexts
            )
            relevancy_result = self.evaluator_relevancy.evaluate(
                query=query,
                response=response_text,
                contexts=contexts
            )
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            return {
                "faithful": None,
                "faithfulness_score": None,
                "relevant": None,
                "relevancy_score": None,
                "error": str(e)
            }
        
        return {
            "faithful": faithfulness_result.passing if faithfulness_result else None,
            "faithfulness_score": getattr(faithfulness_result, 'score', None),
            "relevant": relevancy_result.passing if relevancy_result else None,
            "relevancy_score": getattr(relevancy_result, 'score', None),
        }
    
    
    @rate_limit(calls_per_minute=40)
    def evaluate_rag_triad(self, query: str, response_text: str, contexts: List[str]) -> Dict[str, Any]:
            if not self.evaluator_faithfulness or not self.evaluator_relevancy:
                self._init_evaluators()
            
            contexts_used = 0
            for context in contexts:
                context_sentences = context.split('.')[:2]
                if any(sentence in response_text for sentence in context_sentences):
                    contexts_used += 1
            
            utilization_rate = contexts_used / len(contexts) if contexts else 0
            
            faithfulness_result = self.evaluator_faithfulness.evaluate(
                query=query,
                response=response_text,
                contexts=contexts
            ) if self.evaluator_faithfulness else None
            
            relevancy_result = self.evaluator_relevancy.evaluate(
                query=query,
                response=response_text,
                contexts=contexts
            ) if self.evaluator_relevancy else None
            
            return {
                'faithful': faithfulness_result.passing if faithfulness_result else None,
                'faithfulness_score': getattr(faithfulness_result, 'score', None),
                'relevant': relevancy_result.passing if relevancy_result else None,
                'relevancy_score': getattr(relevancy_result, 'score', None),
                'context_utilization_rate': utilization_rate,
                'contexts_used': contexts_used,
                'contexts_provided': len(contexts)
            }
    
    def _bm25_search(self, query: str, size: int, course_context: Optional[str]) -> List[Dict]:
        mm_query = {
            "multi_match": {
                "query": query,
                "fields": [
                    f"question^{self.settings.get('boost_question', 1)}",
                    f"text^{self.settings.get('boost_text', 1)}"
                ],
                "type": self.settings.get("bm25_type", "best_fields")
            }
        }
        if "minimum_should_match" in self.settings:
            mm_query["multi_match"]["minimum_should_match"] = self.settings["minimum_should_match"]
        if course_context:
            final_query = {"bool": {"must": mm_query, "filter": {"term": {"course": course_context}}}}
        else:
            final_query = mm_query
        try:
            response = self.es_client.search(index=self.index_name, query=final_query, size=size)
            return response.get('hits', {}).get('hits', [])
        except Exception:
            logger.error(f"BM25 search failed: {traceback.format_exc()}")
            return []

    def _vector_search(self, query: str, size: int, course_context: Optional[str]) -> List[Dict]:
        query_vector = self._get_query_embedding(query)
        if not query_vector:
            return []
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                    "params": {"query_vector": query_vector}
                }
            }
        }
        if course_context:
            script_query["script_score"]["query"] = {"term": {"course": course_context}}
        try:
            response = self.es_client.search(index=self.index_name, query=script_query, size=size)
            return response.get('hits', {}).get('hits', [])
        except Exception:
            logger.error(f"Vector search failed: {traceback.format_exc()}")
            return []

    def _hybrid_search(self, query: str, size: int, course_context: Optional[str]) -> List[Dict]:
        bm25_hits = self._bm25_search(query, size * 2, course_context)
        vector_hits = self._vector_search(query, size * 2, course_context)
        scores = {}
        for rank, hit in enumerate(bm25_hits):
            scores[hit['_id']] = scores.get(hit['_id'], 0) + 1.0 / (60 + rank + 1)
        for rank, hit in enumerate(vector_hits):
            scores[hit['_id']] = scores.get(hit['_id'], 0) + 1.0 / (60 + rank + 1)
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:size]
        final_hits = []
        for doc_id in sorted_ids:
            for hit in bm25_hits + vector_hits:
                if hit['_id'] == doc_id:
                    final_hits.append(hit)
                    break
        return final_hits

    def batch_search_faq(self, queries: List[str], k: int, course_context: str = None) -> List[List[Dict]]:
        try:
            batch_size = len(queries)
            logger.info(f"Batch searching {batch_size} queries with k={k}")
            
            search_type = self.settings.get("search_type", "bm25")
            if search_type == "bm25":
                return self._batch_bm25_search(queries, k, course_context)
            elif search_type == "vector":
                return self._batch_vector_search(queries, k, course_context)
            elif search_type == "hybrid":
                return self._batch_hybrid_search(queries, k, course_context)
            else:
                raise ValueError(f"Unknown search_type: {search_type}")
        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            raise

    def _batch_bm25_search(self, queries: List[str], k: int, course_context: str = None) -> List[List[Dict]]:
        from elasticsearch.helpers import bulk
        
        search_body = []
        for query in queries:
            must_conditions = [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^20", "text^1", "course^10"],
                        "type": "best_fields"
                    }
                }
            ]
            
            if course_context:
                must_conditions.append({
                    "term": {"course": course_context}
                })
            
            search_body.append({
                "index": self.index_name
            })
            search_body.append({
                "size": k,
                "query": {
                    "bool": {
                        "must": must_conditions
                    }
                }
            })
        
        response = self.es_client.msearch(body=search_body)
        
        results = []
        for resp in response['responses']:
            hits = resp['hits']['hits'] if 'hits' in resp and 'hits' in resp['hits'] else []
            results.append(hits)
        
        return results

    def _batch_vector_search(self, queries: List[str], k: int, course_context: str = None) -> List[List[Dict]]:
        embeddings = []
        for query in queries:
            embedding = self.embedding_model.encode(query)
            embeddings.append(embedding.tolist())
        
        search_body = []
        for query, embedding in zip(queries, embeddings):
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                        "params": {"query_vector": embedding}
                    }
                }
            }
            
            if course_context:
                script_query["script_score"]["query"] = {
                    "bool": {
                        "filter": {"term": {"course": course_context}}
                    }
                }
            
            search_body.append({
                "index": self.index_name
            })
            search_body.append({
                "size": k,
                "query": script_query
            })
        
        response = self.es_client.msearch(body=search_body)
        
        results = []
        for resp in response['responses']:
            hits = resp['hits']['hits'] if 'hits' in resp and 'hits' in resp['hits'] else []
            results.append(hits)
        
        return results

    def _batch_hybrid_search(self, queries: List[str], k: int, course_context: str = None) -> List[List[Dict]]:
        embeddings = []
        for query in queries:
            embedding = self.embedding_model.encode(query)
            embeddings.append(embedding.tolist())
        
        search_body = []
        for query, embedding in zip(queries, embeddings):
            should_conditions = [
                {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^20", "text^1", "course^10"],
                        "boost": 1.0
                    }
                },
                {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'question_vector') + 1.0",
                            "params": {"query_vector": embedding}
                        },
                        "boost": 1.0
                    }
                }
            ]
            
            filter_conditions = []
            if course_context:
                filter_conditions.append({"term": {"course": course_context}})
            
            query_obj = {
                "size": k,
                "query": {
                    "bool": {
                        "should": should_conditions,
                        "filter": filter_conditions if filter_conditions else None
                    }
                }
            }
            
            if not filter_conditions:
                del query_obj["query"]["bool"]["filter"]
            
            search_body.append({
                "index": self.index_name
            })
            search_body.append(query_obj)
        
        response = self.es_client.msearch(body=search_body)
        
        results = []
        for resp in response['responses']:
            hits = resp['hits']['hits'] if 'hits' in resp and 'hits' in resp['hits'] else []
            results.append(hits)
        
        return results
