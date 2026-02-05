import numpy as np
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any

class Retriever:
    def __init__(self):
        self.bm25 = None
        self.items = [] # Keep reference to items to map index back to ID
        self.id_map = {} # index -> item_id
        
    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def index(self, items: List[Dict[str, Any]]):
        """
        Builds the BM25 index from valid items.
        """
        self.items = items
        self.id_map = {i: item["id"] for i, item in enumerate(items)}
        
        corpus = [self._tokenize(item["title"]) for item in items]
        self.bm25 = BM25Okapi(corpus)
        print(f"Retriever indexed {len(items)} items.")

    def search(self, query: str, k: int = 100) -> List[Dict[str, Any]]:
        """
        Returns top-K items matching the query.
        """
        if not self.bm25:
            return []
            
        tokenized_query = self._tokenize(query)
        # BM25Okapi get_top_n returns the documents (items)
        # But we stored tokenized docs in BM25? No, BM25 stores stats.
        # We need to get scores and sort manually to be efficient or use get_top_n
        
        scores = self.bm25.get_scores(tokenized_query)
        # Get top K indices
        top_n_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_n_indices:
            # Filter zero scores if wanted, but BM25 can be negative? No, usually positive.
            if scores[idx] > 0:
                item = self.items[idx].copy()
                item["score"] = float(scores[idx]) # Add retrieval score
                results.append(item)
                
        return results
