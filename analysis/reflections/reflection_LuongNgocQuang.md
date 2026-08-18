# Project: Production RAG Pipeline

## Part 1: Mapping bài giảng (10 phút)

| Lecture Concept | Module | Hàm cụ thể | Observation |
|----------------|--------|-------------|-------------|
| Semantic chunking | M1 | `chunk_semantic()` | "Threshold 0.85 tạo X chunks vs basic Y chunks" |
| BM25 + Dense fusion | M2 | `reciprocal_rank_fusion()` | "RRF giải quyết..." |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | "Latency Xms, precision..." |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | "Metric X thấp nhất vì..." |
| Contextual embeddings | M5 | `contextual_prepend()` | "Giảm retrieval failure bằng..." |

## Part 2: Khó khăn & giải quyết (10 phút)

- Lỗi gặp phải: `ModuleNotFoundError: No module named 'sentence_transformers'` khi chạy test
- Cách debug: Cài đặt các package cần thiết thông qua pip install
- Kiến thức thiếu: Hiểu rõ hơn về cách sử dụng các thư viện như sentence-transformers, rank-bm25, qdrant-client

## Part 3: Action Plan cho project (10 phút)

### Hiện tại
- RAG pipeline hiện tại: Đã implement đầy đủ 5 modules theo yêu cầu
- Known issues: Một số dependency chưa được cài đặt hoàn chỉnh

### Plan áp dụng
1. [x] Chunking strategy: Semantic chunking với threshold 0.85
2. [x] Search: Hybrid search với BM25 + Dense + RRF
3. [x] Reranking: CrossEncoder reranking với bge-reranker-v2-m3
4. [x] Evaluation: RAGAS với 4 metrics
5. [x] Enrichment: Combined single call mode

### Timeline
- Tuần 1: Cài đặt môi trường và hoàn thiện các module cơ bản
- Tuần 2: Tối ưu hóa hiệu năng và kiểm tra tính chính xác
- Tuần 3: Thực hiện đánh giá và phân tích failure