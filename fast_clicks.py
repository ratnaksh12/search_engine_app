import json
import random
import time
import os

NUM_CLICKS = 20_000
OUTPUT_DIR = "."

def generate_fast_clicks():
    print("Loading items...")
    with open(os.path.join(OUTPUT_DIR, "items.jsonl"), "r") as f:
        items = [json.loads(line) for line in f]
        
    print(f"Generating {NUM_CLICKS} clicks (fast mode)...")
    logs = []
    
    for _ in range(NUM_CLICKS):
        # 1. Pick a "relevant" item
        item = random.choice(items)
        
        # 2. explicit relevance: construct query from title parts
        title_tokens = item["title"].split()
        if not title_tokens: continue
        
        # Pick 1 or 2 tokens
        if len(title_tokens) > 1 and random.random() > 0.5:
             # phrase
             start = random.randint(0, len(title_tokens)-2)
             query = " ".join(title_tokens[start:start+2])
        else:
             query = random.choice(title_tokens)
             
        # 3. Log click
        # Simulate position bias: usually clicked at top
        pos = 0
        r = random.random()
        if r > 0.6: pos = 1
        elif r > 0.8: pos = 2
        elif r > 0.9: pos = random.randint(3, 9)
        
        logs.append({
            "user_id": f"user_{random.randint(1, 1000)}",
            "query": query,
            "item_id": item["id"],
            "position": pos,
            "timestamp": int(time.time()) - random.randint(0, 86400 * 30)
        })

    print("Writing clicks...")
    with open(os.path.join(OUTPUT_DIR, "clicks.jsonl"), "w") as f:
        for click in logs:
            f.write(json.dumps(click) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    generate_fast_clicks()
