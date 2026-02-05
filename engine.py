import json
import os
import threading
import time
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor

from retriever import Retriever
from ranker import Ranker

DATA_DIR = "."
ITEMS_FILE = os.path.join(DATA_DIR, "items.jsonl")
CLICKS_FILE = os.path.join(DATA_DIR, "clicks.jsonl")

class SearchEngine:
    def __init__(self):
        self.retriever = Retriever()
        self.ranker = Ranker()
        self.items = []
        self.lock = threading.Lock()
        self.click_logger = ThreadPoolExecutor(max_workers=1)
        self.query_logs = [] # Store (timestamp, query) for real-time metrics

    def load(self):
        """Loads items and trains models if data exists."""
        print("Loading data...")
        if os.path.exists(ITEMS_FILE):
            with open(ITEMS_FILE, "r") as f:
                self.items = [json.loads(line) for line in f]
            
            # Build Index
            self.retriever.index(self.items)
            
            # Train Ranker if clicks exist
            if os.path.exists(CLICKS_FILE):
                self._train_ranker()
        else:
            print(f"Warning: {ITEMS_FILE} not found. System starts empty.")

    def _train_ranker(self):
        print("Training ranker...")
        clicks = []
        try:
            with open(CLICKS_FILE, "r") as f:
                for line in f:
                    try:
                        clicks.append(json.loads(line))
                    except:
                        continue
        except Exception as e:
            print(f"Error reading clicks: {e}")
            return

        if clicks:
            self.ranker.train(clicks, self.items)

    def search(self, query: str, k: int = 20, user_id: Optional[str] = None) -> Dict:
        start_time = time.time()
        
        # 1. Retrieval (Recall)
        # Fetch more candidates for re-ranking (e.g., 5x K)
        candidates = self.retriever.search(query, k=k*5)
        
        # 2. Ranking (Precision)
        ranked_results = self.ranker.predict(candidates, query)
        
        # 3. Top-K
        final_results = ranked_results[:k]
        
        # 4. Log query for real-time metrics
        with self.lock:
            self.query_logs.append((time.time(), query))
        
        latency_ms = (time.time() - start_time) * 1000
        return {
            "items": final_results,
            "meta": {
                "total_candidates": len(candidates),
                "latency_ms": round(latency_ms, 2)
            }
        }

    def log_click(self, click_data: Dict):
        """
        Async click logging to avoid blocking response.
        """
        self.click_logger.submit(self._append_click_log, click_data)

    def _append_click_log(self, data: Dict):
        with self.lock:
            with open(CLICKS_FILE, "a") as f:
                f.write(json.dumps(data) + "\n")

    def add_items(self, new_items: List[Dict]):
        """
        Bulk add items. Simple implementation: Append and Re-index.
        Real production would use incremental index.
        """
        with self.lock:
            with open(ITEMS_FILE, "a") as f:
                for item in new_items:
                    f.write(json.dumps(item) + "\n")
            
            self.items.extend(new_items)
            # Full Re-index (Blocking!) - Acceptable for "Mini" system
            self.retriever.index(self.items)

    def reindex(self):
        """Manually trigger re-indexing and ranker training."""
        with self.lock:
            self.retriever.index(self.items)
            self._train_ranker()

    def get_top_queries(self, window_seconds: int = 300):
        now = time.time()
        with self.lock:
            # Cleanup old logs
            self.query_logs = [ql for ql in self.query_logs if now - ql[0] <= window_seconds]
            
            # Simple count
            counts = {}
            for _, q in self.query_logs:
                counts[q] = counts.get(q, 0) + 1
            
            # Sort and return
            sorted_q = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            return [{"query": q, "count": c} for q, c in sorted_q[:10]]

    def get_stats(self):
        return {
            "items_count": len(self.items),
            "has_ranker": self.ranker.model is not None
        }
