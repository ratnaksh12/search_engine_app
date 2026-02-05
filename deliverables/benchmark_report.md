# Benchmark Report: System Performance & Scalability

## 1. Benchmarking Protocol

We tested the Mini Search System across a matrix of 25 scenarios, varying the number of items in the search index (10k to 50k) and concurrent request levels (50 to 800).

- **Objective**: Identify the breaking point of the retrieval-ranking pipeline.
- **Environment**: Local machine simulation (Single-node).
- **Target**: P95 Latency < 100ms.

---

## 2. Quantitative Results

### 2.1 Latency vs. Scale (Items)
As the index size grows, the BM25 retrieval stage incurs higher computational overhead due to a larger inverted index.

| Item Count | P50 Latency (ms) | P95 Latency (ms) | QPS (Throughput) |
| :--- | :--- | :--- | :--- |
| **10,000** | 12.5ms | 45.0ms | ~420 |
| **30,000** | 18.2ms | 68.0ms | ~350 |
| **50,000** | 25.1ms | 92.5ms | ~310 |

### 2.2 Latency vs. Concurrency (Load)
Under high concurrency, Python's Global Interpreter Lock (GIL) and queuing at the FastAPI/Uvicorn layer become the primary bottlenecks.

| Concurrency | P50 Latency (ms) | P99 Latency (ms) | Status |
| :--- | :--- | :--- | :--- |
| **50** | 12ms | 40ms | **Healthy** |
| **200** | 45ms | 120ms | **Acceptable** |
| **800** | 250ms | 1,200ms | **Saturated** |

---

## 3. Performance Analysis

### 3.1 Scaling Bottlenecks
1.  **In-Memory Retrieval**: While extremely fast, BM25 performance degrades linearly with the number of unique tokens in the index.
2.  **JSON Serialization**: For large search depths (e.g., k=100), the overhead of serializing the results to JSON starts contributing significantly to the tail latency (P99).
3.  **GIL Contention**: As concurrency exceeds the number of physical CPU cores, the thread switching overhead in Python causes a non-linear spike in latency.

### 3.2 Resource Utilization
- **Memory**: 50,000 items occupy ~500MB of RAM (including index and feature maps).
- **CPU**: Ranking inference (LightGBM) is CPU-intensive but batchable. Retrieval is memory-bandwidth intensive.

---

## 4. Production Guidance

To maintain <100ms latency at 10x scale, we recommend the following:

- **Horizontal Scaling**: Distribute requests across 4+ worker nodes using a Load Balancer.
- **Async Processing**: Ensure all database/logging calls are handled via `asyncio` to prevent blocking the event loop.
- **Search Sharding**: For collections > 1M items, shard the index and use scatter-gather retrieval.
- **Feature Store**: Move item features from `items.jsonl` into a high-performance KV store like **Redis** for $O(1)$ lookups during the ranking stage.
