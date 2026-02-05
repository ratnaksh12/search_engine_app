# Design Document: Mini Search System (Production Grade)

## 1. Executive Summary

This document outlines the architecture, design decisions, and performance characteristics of the "Mini Search System," a high-performance two-stage retrieval and ranking pipeline designed for e-commerce search. The system scales to 50k items with sub-100ms P95 latency and provides a framework for real-time feedback loop integration.

## 2. Architecture: Two-Stage Pipeline

The system adopts the industry-standard multi-stage approach to balance retrieval recall with ranking precision.

### 2.1 Stage 1: Retrieval (Candidate Generation)
- **Engine**: In-memory BM25 (Best Matching 25) implementation.
- **Goal**: Efficiently filter the total corpus (50k+ items) down to the top 100 most relevant candidates based on lexical similarity.
- **Process**:
    1. **Tokenization**: Standard whitespace and punctuation stripping.
    2. **Inverted Index**: Maps tokens to item IDs and term frequencies.
    3. **Scoring**: Calculates $score(D, Q) = \sum_{q_i \in Q} IDF(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot (1 - b + b \cdot \frac{|D|}{avgdl})}$.
- **Rationale**: Lexical retrieval is computationally cheap and ensures high recall for keyword-based queries.

### 2.2 Stage 2: Ranking (Precision Refinement)
- **Engine**: LightGBM LambdaRank.
- **Goal**: Re-order the top 100 candidates using a rich set of features that BM25 ignores.
- **Loss Function**: NDCG optimization via LambdaMART.
- **Features**:
    - **Lexical**: Exact title overlap, BM25 score.
    - **Static Content**: Item Price, Quality Score (derived from attributes).
    - **Engagement**: Item Popularity (historical click counts).
- **Position Bias Handling**: During training, we apply **Inverse Propensity Weighting (IPW)**. Clicks at higher positions are down-weighted compared to clicks at lower positions to compensate for the higher examination probability of top-ranked items.

## 3. Component Deep Dive

### 3.1 Retriever Component (`retriever.py`)
The retriever maintains an in-memory dictionary acting as an inverted index. 
- **Indexing**: O(N * L) where N is number of items and L is average title length. 
- **Querying**: O(Q * D) where Q is query tokens and D is document frequency of tokens.

### 3.2 Ranker Component (`ranker.py`)
- **Initialization**: Loads a pre-trained LightGBM Booster.
- **Feature Extraction**: Real-time join between candidate IDs and their metadata (price, quality).
- **Inference**: Batch prediction on 100 candidates takes ~10-15ms.

### 3.3 API Layer (`main.py` / `engine.py`)
- **Framework**: FastAPI with Uvicorn.
- **Endpoints**:
    - `GET /search`: Unified retrieval + ranking flow.
    - `POST /feedback/click`: Asynchronous logging of user activity to `clicks.jsonl`.
    - `POST /items/bulk`: Dynamic indexing of new items.

## 4. Performance & Latency Budget

Target P95 Latency: **< 100ms**

| Component | P50 (ms) | P95 (ms) | Bottleneck |
| :--- | :--- | :--- | :--- |
| **Retrieval** | 10ms | 25ms | Global Inverted Index size |
| **Feature Extraction** | 5ms | 12ms | JSON lookups / Join |
| **Ranking (ML)** | 12ms | 20ms | Tree depth / Candidate count |
| **IO / FastAPI** | 3ms | 10ms | JSON serialization |
| **Total** | **~30ms** | **~67ms** | |

## 5. Scaling Strategy (10x - 100x Growth)

### 5.1 To 500k Items (10x)
- **Vector Search**: Transition from BM25 to HNSW (Hierarchical Navigable Small World) for semantic retrieval.
- **Memory Management**: Move from in-memory dicts to a persistent KV store (Redis) for features.

### 5.2 To 1000+ QPS
- **Concurrency**: python-based systems are limited by the GIL. We recommend horizontal scaling via Kubernetes and distributing the index across shards.
- **Result Caching**: Implement Redis-based LRU caching for high-frequency queries ("laptop", "phone").

## 6. Evaluation & Continuous Improvement

### 6.1 Metrics
- **Offline**: NDCG@10 (Normalized Discounted Cumulative Gain) and MRR (Mean Reciprocal Rank).
- **Online (Simulated)**: CTR (Click-Through Rate) monitoring.

### 6.2 The Feedback Loop
The system is designed for **Exploration-Exploitation**. Periodically, the ranker model is retrained on `clicks.jsonl` using the latest user feedback to adapt to seasonal trends or shifts in item popularity.

## 7. Operational Roadmap
1. **Version 1.1**: Add a dedicated `/clear` endpoint for index management.
2. **Version 1.2**: Implement multi-threading for feature extraction.
3. **Version 2.0**: Migrating to a vector database (Pinecone/Milvus) for hybrid search support (Keyword + Embedding).
