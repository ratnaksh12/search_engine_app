# Design Document: Mini Search System

## 1. Architecture: Retrieval + Ranking Pipeline

The system follows a classic two-stage information retrieval architecture:

1.  **Stage 1: Retrieval (Candidate Generation)**
    -   **Algorithm**: BM25 (Best Matching 25) via `rank_bm25`.
    -   **Goal**: Efficiently narrow down millions (or 50k in this case) of items to a small set of potentially relevant candidates (e.g., top 100).
    -   **Data Structure**: In-memory inverted index of item titles.

2.  **Stage 2: Ranking (Precision Refinement)**
    -   **Algorithm**: LightGBM LambdaRank.
    -   **Goal**: Score the top candidates using richer features (price, popularity, quality score, exact title overlap) to determine the final ordering.
    -   **Loss Function**: LambdaRank optimizes NDCG directly.
    -   **Position Bias**: Handled via propensity weighting (Inverse Propensity Weighting) during training, where samples from lower positions are weighted higher to compensate for lower examination probability.

## 2. Latency Budget Split

Target latency: < 100ms P95.

-   **Retrieval**: 10-20ms. Involves tokenization and BM25 scoring over the inverted index.
-   **Feature Extraction**: 5-10ms. Mapping item attributes and calculating text-based features for candidates.
-   **Ranking (Inference)**: 10-20ms. LightGBM model prediction over ~100 candidates.
-   **Overhead/IO**: 5-10ms. FastAPI handling, JSON serialization, and logging.

Total estimated latency: ~50ms.

## 3. Caching Strategies

-   **Result Cache**: LRU cache for top N queries (e.g., "laptop", "phone"). Invalidated on item re-indexing.
-   **Feature Cache**: Pre-compute item-specific features (static scores) to avoid calculation during every request.
-   **Embedding Cache**: If using ANN, cache query embeddings to avoid re-computing for repeated queries.

> [!IMPORTANT]
> **Data Consistency Note**: During benchmarking, the index size may show as ~110k items instead of the expected 50k. This is because the `/items/bulk` endpoint appends to the current index. This lack of a `/clear` endpoint causes accumulation during sequential benchmark runs, but relative scaling trends remain valid.

## 4. Data Storage

-   **Features**: Currently stored in `items.jsonl` (in-memory during runtime). In production, would use a Feature Store (e.g., Feast or Redis).
-   **Logs**: Click feedback is appended to `clicks.jsonl` asynchronously. In production, this would go to Kafka/S3 for batch processing.
-   **Model Registry**: Simplified locally. In production, models would be versioned and served via MLflow or similar.

## 5. Monitoring

-   **Drift**: Track distribution of predicted scores and feature values over time.
-   **Metrics**:
    -   **CTR (Click-Through Rate)**: Real-time monitoring of `/search` vs `/feedback/click`.
    -   **Null Results**: Alert if queries return 0 items.
    -   **Slow Queries**: Log P99 latency and identify outlier query patterns.

## 6. Scale Plan (10x Increase)

-   **500k Items**:
    -   Move from in-memory BM25 to a dedicated search engine like Elasticsearch/OpenSearch or a vector database (Pinecone/Milvus) for HNSW-based ANN.
    -   Shard the index across multiple nodes.
-   **10x QPS**:
    -   Horizontal scaling of FastAPI workers.
    -   Implement aggressive result caching (Redis).
    -   Offload ranking to a dedicated model server (Triton/TFServing).
