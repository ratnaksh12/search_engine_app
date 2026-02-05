import lightgbm as lgb
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Any

class Ranker:
    def __init__(self):
        self.model = None
        self.feature_cols = ["price", "popularity", "quality", "title_overlap"]
        
    def _extract_features(self, item: Dict[str, Any], query: str) -> List[float]:
        q_tokens = set(query.lower().split())
        t_tokens = set(item["title"].lower().split())
        overlap = len(q_tokens.intersection(t_tokens))
        
        return [
            float(item.get("price", 0.0)),
            float(item.get("features", {}).get("popularity", 0.0)),
            float(item.get("features", {}).get("quality_score", 0.0)),
            float(overlap)
        ]

    def prepare_data(self, clicks: List[Dict], items_map: Dict[str, Dict]):
        """
        Prepare X, y, group, and weights for LightGBM LambdaRank.
        Includes propensity weighting for position bias.
        """
        X = []
        y = []
        groups = []
        weights = []
        
        # Group clicks by query
        query_groups = {}
        for c in clicks:
            q = c["query"]
            if q not in query_groups:
                query_groups[q] = []
            query_groups[q].append(c)
            
        all_item_ids = list(items_map.keys())
        
        for q, query_clicks in query_groups.items():
            current_group_size = 0
            pos_item_ids = set([c["item_id"] for c in query_clicks])
            
            # Positives
            for click in query_clicks:
                pid = click["item_id"]
                if pid not in items_map: continue
                feat = self._extract_features(items_map[pid], q)
                X.append(feat)
                y.append(1)
                
                # Propensity Weighting: w = 1 / P(examine | position)
                # Using simple 1 / log2(pos + 2) as inverse propensity
                pos = click.get("position", 0)
                propensity = 1.0 / np.log2(pos + 2)
                weights.append(1.0 / propensity) # Inverse propensity weight
                
                current_group_size += 1
                
            # Negatives (Sample 5x query sessions)
            num_neg = len(query_clicks) * 5
            neg_ids = random.sample(all_item_ids, min(len(all_item_ids), num_neg))
            
            for nid in neg_ids:
                if nid in pos_item_ids: continue
                if nid not in items_map: continue
                feat = self._extract_features(items_map[nid], q)
                X.append(feat)
                y.append(0)
                weights.append(1.0) # Assume uniform propensity for negatives (or baseline)
                current_group_size += 1
                
            groups.append(current_group_size)
            
        return np.array(X), np.array(y), np.array(groups), np.array(weights)

    def train(self, clicks: List[Dict], items: List[Dict]):
        print(f"Training Ranker with {len(clicks)} clicks...")
        items_map = {i["id"]: i for i in items}
        
        X, y, group, sample_weight = self.prepare_data(clicks, items_map)
        
        if len(X) == 0:
            print("No training data found.")
            return

        gbm = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=100,
            learning_rate=0.1
        )
        
        gbm.fit(X, y, group=group, sample_weight=sample_weight)
        self.model = gbm
        print("Ranker training complete.")

    def predict(self, candidates: List[Dict], query: str) -> List[Dict]:
        """
        Re-rank candidates.
        """
        if not self.model or not candidates:
            return candidates
            
        X_pred = [self._extract_features(item, query) for item in candidates]
        scores = self.model.predict(X_pred)
        
        # Attach scores and sort
        for i, item in enumerate(candidates):
            item["ranker_score"] = float(scores[i])
            
        # Sort by ranker_score descending
        ranked = sorted(candidates, key=lambda x: x["ranker_score"], reverse=True)
        return ranked
