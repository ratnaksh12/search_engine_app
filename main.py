from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import time
from engine import SearchEngine

app = FastAPI(title="Mini Search System")
engine = SearchEngine()

class SearchRequest(BaseModel):
    q: str
    k: int = 20
    user_id: Optional[str] = None

class ClickFeedback(BaseModel):
    user_id: str
    query: str
    item_id: str
    position: int
    ts: Optional[int] = None

@app.on_event("startup")
def startup_event():
    engine.load()

@app.get("/")
def read_root():
    return {"status": "ok", "stats": engine.get_stats()}

@app.get("/search")
def search(q: str, k: int = 20, user_id: Optional[str] = None):
    results = engine.search(q, k, user_id)
    return results

@app.post("/feedback/click")
def report_click(feedback: ClickFeedback):
    if feedback.ts is None:
        feedback.ts = int(time.time())
    
    engine.log_click(feedback.dict())
    return {"status": "accepted"}

@app.post("/items/bulk")
def add_items(items: List[Dict[str, Any]]):
    engine.add_items(items)
    return {"status": "indexed", "count": len(items)}

@app.get("/top_queries")
def get_top_queries(window: str = "5m"):
    # Parse window like '5m', '10s', '1h'
    seconds = 300
    if window.endswith("m"):
        seconds = int(window[:-1]) * 60
    elif window.endswith("s"):
        seconds = int(window[:-1])
    elif window.endswith("h"):
        seconds = int(window[:-1]) * 3600
        
    return engine.get_top_queries(seconds)

@app.post("/reindex")
def reindex():
    engine.reindex()
    return {"status": "reindexed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
