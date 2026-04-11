"""
Search module — Keyword (BM25) + Semantic (FAISS) + Hybrid search.

Usage:
    from search import SearchEngine
    engine = SearchEngine()
    results = engine.hybrid_search("what is agentic AI", top_k=5)
"""

import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from nlp import CHUNKS_FILE, BM25_FILE, FAISS_FILE, EMBEDDING_MODEL, tokenize


# Minimum cosine similarity threshold — results below this are considered
# irrelevant (i.e., the query is out of the scope of the indexed content).

MIN_RELEVANCE_THRESHOLD = 0.0


class SearchEngine:
    """Hybrid search engine combining BM25 keyword search and FAISS semantic search."""

    def __init__(self):
        print("Loading search engine...")

        # Load chunks
        with open(CHUNKS_FILE, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # Load BM25 index
        with open(BM25_FILE, "rb") as f:
            self.bm25 = pickle.load(f)

        # Load FAISS index
        self.faiss_index = faiss.read_index(FAISS_FILE)

        # Load embedding model
        self.model = SentenceTransformer(EMBEDDING_MODEL)

        # Initialize model attribute for lazy loading
        # self.model = None

        print(f" Loaded {len(self.chunks)} chunks, "
              f"FAISS index ({self.faiss_index.ntotal} vectors)")
    def load_model(self):
        """ function for using the embedding model as lazy loading"""
        if self.model is None:
            self.model = SentenceTransformer(EMBEDDING_MODEL)
        return self.model
    def keyword_search(self, query, top_k=10):
        """BM25 keyword search."""
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    **self.chunks[idx],
                    "score": float(scores[idx]),
                    "method": "keyword",
                })

        return results

    def semantic_search(self, query, top_k=5):
        """FAISS semantic search using embeddings."""
        # model = self.load_model()
        query_embedding = self.model.encode(
            [query], normalize_embeddings=True
        ).astype("float32")

        scores, indices = self.faiss_index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score > MIN_RELEVANCE_THRESHOLD:
                results.append({
                    **self.chunks[idx],
                    "score": float(score),
                    "method": "semantic",
                })

        return results

    def hybrid_search(self, query, top_k=10, alpha=0.5):
        """
        Hybrid search combining keyword and semantic results.

        Args:
            query: Search query string
            top_k: Number of results to return
            alpha: Weight for semantic search (1-alpha for keyword)
                   alpha=1.0 → pure semantic, alpha=0.0 → pure keyword
        """
        # Get more results from each method for better merging

        keyword_results = self.keyword_search(query, top_k=top_k * 3)
        # print(len(keyword_results))
        semantic_results = self.semantic_search(query, top_k=top_k * 3)
        # print(semantic_results)

        # Save raw semantic scores before normalization (for threshold check)
        raw_semantic_scores = {r["chunk_id"]: r["score"] for r in semantic_results}

        # Normalize scores to [0, 1]
        keyword_results = self._normalize_scores(keyword_results)
        semantic_results = self._normalize_scores(semantic_results)

        # Merge results by chunk_id
        merged = {}

        for r in keyword_results:
            cid = r["chunk_id"]
            merged[cid] = {
                **r,
                "keyword_score": r["score"],
                "semantic_score": 0.0,
                "raw_semantic_score": raw_semantic_scores.get(cid, 0.0),
            }

        for r in semantic_results:
            cid = r["chunk_id"]
            if cid in merged:
                merged[cid]["semantic_score"] = r["score"]
                merged[cid]["raw_semantic_score"] = raw_semantic_scores.get(cid, 0.0)
            else:
                merged[cid] = {
                    **r,
                    "keyword_score": 0.0,
                    "semantic_score": r["score"],
                    "raw_semantic_score": raw_semantic_scores.get(cid, 0.0),
                }

        # Calculate hybrid score
        for cid, r in merged.items():
            r["score"] = alpha * r["semantic_score"] + (1 - alpha) * r["keyword_score"]
            r["method"] = "hybrid"

        # Sort by hybrid score and deduplicate by URL (keep best chunk per blog)
        sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

        seen_urls = set()
        deduplicated = []
        for r in sorted_results:
            if r["url"] not in seen_urls:
                seen_urls.add(r["url"])
                deduplicated.append(r)
            if len(deduplicated) >= top_k:
                break

        return deduplicated

    @staticmethod
    def _normalize_scores(results):
        """Normalize scores to [0, 1] range."""
        if not results:
            return results

        scores = [r["score"] for r in results]
        min_s, max_s = min(scores), max(scores)

        if max_s - min_s == 0:
            for r in results:
                r["score"] = 1.0
        else:
            for r in results:
                r["score"] = (r["score"] - min_s) / (max_s - min_s)

        return results

    def get_stats(self):
        """Return index statistics."""
        unique_blogs = len(set(c["url"] for c in self.chunks))
        return {
            "total_chunks": len(self.chunks),
            "total_blogs": unique_blogs,
            "faiss_vectors": self.faiss_index.ntotal,
            "embedding_model": EMBEDDING_MODEL,
        }
