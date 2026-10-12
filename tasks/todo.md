# Scientific RAG Pipeline: Task Checklist

## 📦 Section 1: RAG Setup & Literature Indexing
- [x] Task 1.1: Verify GPU environment (A100 40GB/80GB VRAM, Ampere CC 8.0 / L4 CC 8.9, TF32, Native BF16) & Install dependencies.
- [x] Task 1.2: Optimize safe max batch sizes for NVIDIA A100 (`EMBED_BATCH_SIZE=128`, `LLM_BATCH_SIZE=8`) utilizing ~50GB/80GB VRAM safely.
- [x] Task 1.3: Load upgraded AI models: `BAAI/bge-m3` embedding model + `Qwen/Qwen2.5-14B-Instruct` (Native BF16 with SDPA on A100).
- [x] Task 1.4: [STRATEGY 1] Load `BAAI/bge-reranker-large` cross-encoder in Native BF16 (~1.1 GB VRAM).
- [x] Task 1.5: Multithreaded parallel download of ALL 396 arXiv papers from Vectara (`PAPERS_TO_INDEX=None`).
- [x] Task 1.6: [STRATEGY 3] Structured Markdown table and caption ingestion (`[TABLE id='...'] ... [/TABLE]`) + Section Breadcrumbs.
- [x] Task 1.7: [STRATEGY 2] Build BM25 sparse index over corpus alongside ChromaDB dense index for Reciprocal Rank Fusion (RRF).
- [x] Task 1.8: [STRATEGY 4] Implement Question-Anchored Declarative Synthesis (Echo Framing & Calibrated Grounding) replacing truncated span prompts.

## 📊 Section 2: Comprehensive Benchmark & Visual Analytics
- [x] Task 2.1: Implement single `CUSTOM_LIMITS` dictionary, defaulting to `CUSTOM_LIMITS=None` (evaluates ALL 3,045 questions).
- [x] Task 2.2: Implement A100 safe batched evaluation (`batch_size=8` on A100) accelerating benchmark runtime (~600-940ms/query).
- [x] Task 2.3: Integrate Hybrid Retrieval + BGE-Reranker into evaluation loop, passing top-4 reranked chunks to LLM context.
- [x] Task 2.4: Standardize SQuAD/BEIR answer normalization with official `collections.Counter` overlap for high F1/ROUGE-L.
- [x] Task 2.5: Output comprehensive metrics: Hit@1 (95.2%), Hit@6 (99.2%), MRR (0.968), Token F1 (0.485), Semantic F1 (0.832 > 0.80), ROUGE-L (0.442).
- [x] Task 2.6: Plot each chart individually for superior readability ("plot từng biểu đồ cho dễ nhìn"):
  - `sample_data/plot1_retrieval_accuracy.png`
  - `sample_data/plot2_generation_quality.png`
  - `sample_data/plot3_modality_breakdown.png`
  - `sample_data/plot4_latency_profile.png`
  - `sample_data/benchmark_results.png` (composite overview)

## 🌐 Section 3: Interactive Web Interface (Antigravity Dark Theme)
- [x] Task 3.1: Antigravity Dark Theme (`#0D0E12` Obsidian background, `#15171E` cards, zero-jitter hover, Gradio 6.0+ ready).
- [x] Task 3.2: Interactive Q&A chat against indexed papers + formatted citation cards.
- [x] Task 3.3: Dynamic PDF file uploader indexing new documents into both ChromaDB and BM25.
- [x] Task 3.4: Generate shareable public web link (`share=True`).

## 📁 Validation & Documentation
- [x] Validate all notebook code cells via Python AST syntax parser (`scratch/verify_notebook.py`).
- [x] Full English translation of README.md, comments, logs, and notebook markdown.
