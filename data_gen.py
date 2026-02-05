import random
import json
import time
import numpy as np
import os
from typing import List, Dict

# Configuration
NUM_ITEMS = 50_000
NUM_QUERIES = 100_000 # Pool of queries
NUM_HISTORICAL_CLICKS = 50_000
OUTPUT_DIR = "."

# Vocabulary for synthetic data
BRANDS = ["BrandA", "BrandB", "BrandC", "BrandD", "SuperTech", "MegaCorp", "SoftSoft", "HardWare"]
CATEGORIES = ["Electronics", "Books", "Clothing", "Home", "Garden", "Toys", "Sports", "Automotive"]
ADJECTIVES = ["New", "Used", "Refurbished", "Shiny", "Fast", "Ergonomic", "Premium", "Cheap", "Luxury"]
NOUNS = ["Laptop", "Phone", "Table", "Chair", "Headphones", "Mouse", "Keyboard", "Monitor", "Cable", "Charger", "Shirt", "Pants", "Shoes", "Ball", "Bat", "Racket"]

def generate_items(n=NUM_ITEMS) -> List[Dict]:
    items = []
    print(f"Generating {n} items...")
    for i in range(n):
        cat = random.choice(CATEGORIES)
        brand = random.choice(BRANDS)
        noun = random.choice(NOUNS)
        adj = random.choice(ADJECTIVES)
        
        # Title construction: Brand + Adjective + Noun
        title = f"{brand} {adj} {noun}"
        
        # Add some randomness to title to make it unique-ish
        if random.random() > 0.5:
            title += f" {random.randint(100, 999)}"
            
        item = {
            "id": f"item_{i}",
            "title": title,
            "category": cat,
            "brand": brand,
            "price": round(random.uniform(10.0, 1000.0), 2),
            "description": f"A very {adj.lower()} {noun.lower()} from {brand}. Perfect for your needs.",
            "features": {
                "popularity": round(random.random(), 4),
                "quality_score": round(random.uniform(0.5, 1.0), 4)
            }
        }
        items.append(item)
    return items

def generate_queries(n=NUM_QUERIES) -> List[str]:
    print(f"Generating {n} queries...")
    queries = []
    # Mix of single words (broad) and double words (specific)
    for _ in range(n):
        if random.random() < 0.3:
            q = random.choice(NOUNS)
        elif random.random() < 0.6:
            q = f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}"
        else:
             q = f"{random.choice(BRANDS)} {random.choice(NOUNS)}"
        queries.append(q)
    return queries

def simulate_relevance(query: str, item: Dict) -> float:
    """
    Ground truth relevance function (Oracle).
    Returns probability of relevance (0.0 to 1.0).
    """
    q_tokens = set(query.lower().split())
    title_tokens = set(item["title"].lower().split())
    
    # Exact token match in title is a strong signal
    overlap = len(q_tokens.intersection(title_tokens))
    
    base_rel = 0.0
    if overlap > 0:
        base_rel = 0.2 + (0.3 * overlap) # Max ~0.8
        
    # Category match bonus? (Not used for now, simplistic text match)
    
    # Popularity bias
    pop_bias = item["features"]["popularity"] * 0.1
    
    final_rel = min(1.0, base_rel + pop_bias)
    return final_rel

def generate_click_logs(items: List[Dict], queries: List[str], n_clicks=NUM_HISTORICAL_CLICKS):
    print(f"Generating {n_clicks} historical clicks...")
    
    # Pre-index items by words for faster candidate retrieval in simulation
    # Simple inverted index for generation speed
    inv_index = {}
    for item in items:
        for token in item["title"].lower().split():
            if token not in inv_index:
                inv_index[token] = []
            inv_index[token].append(item)
            
    logs = []
    
    for _ in range(n_clicks):
        user_id = f"user_{random.randint(1, 1000)}"
        query = random.choice(queries)
        
        # Retrieve candidates (simulate a basic search engine)
        q_tokens = query.lower().split()
        candidates = []
        for t in q_tokens:
            if t in inv_index:
                candidates.extend(inv_index[t])
        
        if not candidates:
            continue
            
        # Deduplicate and sample candidates to show to user
        candidates = list({c["id"]: c for c in candidates}.values())
        if len(candidates) > 20:
            candidates = random.sample(candidates, 20)
            
        # Rank candidates randomly initially (simulating poor initial ranker) or by basic match
        random.shuffle(candidates)
        
        # User browses
        for pos, item in enumerate(candidates):
            rel_prob = simulate_relevance(query, item)
            
            # Position Bias: P(examine | pos) ~= 1 / log2(pos + 2)
            exam_prob = 1.0 / np.log2(pos + 2)
            
            # Click = Examine * Relevant (simplified CCM)
            click_prob = exam_prob * rel_prob
            
            if random.random() < click_prob:
                logs.append({
                    "user_id": user_id,
                    "query": query,
                    "item_id": item["id"],
                    "position": pos,
                    "timestamp": int(time.time()) - random.randint(0, 86400 * 30) # Past 30 days
                })
                # Assume user stops after click? Or continues? Let's assume single click per session for simplicity often
                break
                
    return logs

if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    
    # 1. Generate Items
    items = generate_items()
    with open(os.path.join(OUTPUT_DIR, "items.jsonl"), "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")
            
    # 2. Generate Queries
    queries = generate_queries()
    # Save a sample of queries? No need to save queries list per se, but good for sim
    # We'll rely on generating them on fly for sim, or just re-use this list.
    
    # 3. Generate Historical Clicks
    clicks = generate_click_logs(items, queries)
    with open(os.path.join(OUTPUT_DIR, "clicks.jsonl"), "w") as f:
        for click in clicks:
            f.write(json.dumps(click) + "\n")
            
    print(f"Data generation complete.")
    print(f"Items: {len(items)}")
    print(f"Clicks: {len(clicks)}")
