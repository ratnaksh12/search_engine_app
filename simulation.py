import requests
import json
import random
import time
import numpy as np
from data_gen import generate_queries, simulate_relevance, generate_items

API_URL = "http://localhost:8000"

def run_simulation(steps=1000):
    print(f"Starting simulation for {steps} steps...")
    
    # We need ground truth items to simulate user relevance
    # Load items locally to know what the user "wants"
    with open("items.jsonl", "r") as f:
        items = [json.loads(line) for line in f]
    items_map = {i["id"]: i for i in items}
    
    queries = generate_queries(n=100) # Small set of queries to repeat
    
    ctr_history = []
    
    for i in range(steps):
        query = random.choice(queries)
        
        # 1. Search
        try:
            resp = requests.get(f"{API_URL}/search", params={"q": query, "k": 20})
            results = resp.json()["items"]
        except Exception as e:
            print(f"Search failed: {e}")
            time.sleep(1)
            continue
            
        if not results:
            continue
            
        # 2. User Behaviour Simulation
        clicked = False
        for pos, item_res in enumerate(results):
            # Get full item details (the ground truth)
            real_item = items_map.get(item_res["id"])
            if not real_item: continue
            
            rel_prob = simulate_relevance(query, real_item)
            exam_prob = 1.0 / np.log2(pos + 2)
            click_prob = exam_prob * rel_prob
            
            if random.random() < click_prob:
                # Click!
                requests.post(f"{API_URL}/feedback/click", json={
                    "user_id": f"sim_user_{random.randint(1,100)}",
                    "query": query,
                    "item_id": item_res["id"],
                    "position": pos,
                    "ts": int(time.time())
                })
                clicked = True
                break # Single click assumption
        
        ctr_history.append(1 if clicked else 0)
        
        if i % 100 == 0:
            current_ctr = np.mean(ctr_history[-100:])
            print(f"Step {i}: CTR (last 100) = {current_ctr:.4f}")
            
if __name__ == "__main__":
    run_simulation()
