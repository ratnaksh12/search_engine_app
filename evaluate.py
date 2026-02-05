import json
import os
import numpy as np
import pandas as pd
from ranker import Ranker
from retriever import Retriever
from sklearn.model_selection import train_test_split
def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def mrr_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if np.sum(r) == 0:
        return 0.
    return 1. / (np.argmax(r) + 1)

def evaluate_offline(clicks_path="clicks.jsonl", items_path="items.jsonl"):
    print("Loading data for evaluation...")
    with open(items_path, "r") as f:
        items = [json.loads(line) for line in f]
    
    clicks = []
    if os.path.exists(clicks_path):
        with open(clicks_path, "r") as f:
            for line in f:
                clicks.append(json.loads(line))
    
    if not clicks:
        print("No clicks found. Use data_gen.py first.")
        return

    print(f"Total clicks: {len(clicks)}")
    user_ids = list(set([c["user_id"] for c in clicks]))
    train_users, test_users = train_test_split(user_ids, test_size=0.2, random_state=42)
    
    train_clicks = [c for c in clicks if c["user_id"] in train_users]
    test_clicks = [c for c in clicks if c["user_id"] in test_users]
    
    # Train Ranker
    ranker = Ranker()
    ranker.train(train_clicks, items)
    
    # Evaluate
    test_groups = {}
    for c in test_clicks:
        if c["query"] not in test_groups:
            test_groups[c["query"]] = []
        test_groups[c["query"]].append(c["item_id"])
        
    retriever = Retriever()
    retriever.index(items)
    
    metrics = {
        "baseline": {"ndcg": [], "mrr": [], "recall": []},
        "ranker": {"ndcg": [], "mrr": [], "recall": []}
    }
    
    print("Evaluating systems...")
    for query, clicked_ids in list(test_groups.items())[:500]: # Sample for speed
        clicked_set = set(clicked_ids)
        
        # 1. Retriever-only baseline
        candidates = retriever.search(query, k=50)
        
        # Position-aware implicit relevance: Clicked=2, Top5=1, others=0
        rel_baseline = []
        for rank, c in enumerate(candidates[:10]):
            if c["id"] in clicked_set:
                rel_baseline.append(2)
            elif rank < 5:
                rel_baseline.append(1)
            else:
                rel_baseline.append(0)
                
        metrics["baseline"]["ndcg"].append(ndcg_at_k(rel_baseline, 10))
        metrics["baseline"]["mrr"].append(mrr_at_k(rel_baseline, 10))
        
        found_in_recall = any(c["id"] in clicked_set for c in candidates)
        metrics["baseline"]["recall"].append(1 if found_in_recall else 0)
        
        # 2. Retriever + Ranker
        ranked = ranker.predict(candidates, query)
        
        rel_ranker = []
        for rank, c in enumerate(ranked[:10]):
            if c["id"] in clicked_set:
                rel_ranker.append(2)
            elif rank < 5:
                rel_ranker.append(1)
            else:
                rel_ranker.append(0)
                
        metrics["ranker"]["ndcg"].append(ndcg_at_k(rel_ranker, 10))
        metrics["ranker"]["mrr"].append(mrr_at_k(rel_ranker, 10))
        metrics["ranker"]["recall"].append(1 if found_in_recall else 0) # Recall is same as candidates
        
    print("\nOffline Evaluation Results:")
    print("-" * 40)
    summary_lines = ["Offline Evaluation Results:", "-" * 40]
    for sys_name in ["baseline", "ranker"]:
        line1 = f"System: {sys_name}"
        line2 = f"  NDCG@10:  {np.mean(metrics[sys_name]['ndcg']):.4f}"
        line3 = f"  MRR@10:   {np.mean(metrics[sys_name]['mrr']):.4f}"
        line4 = f"  Recall@50: {np.mean(metrics[sys_name]['recall']):.4f}"
        print(line1)
        print(line2)
        print(line3)
        print(line4)
        print("-" * 40)
        summary_lines.extend([line1, line2, line3, line4, "-" * 40])
        
    with open("offline_metrics_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

if __name__ == "__main__":
    evaluate_offline()
