# Implementation Plan: High-Performance Multilingual RAG Pipeline on Google Colab (L4 GPU)

## Overview
This plan specifies an end-to-end, high-performance Retrieval-Augmented Generation (RAG) system running entirely inside Google Colab, optimized for **NVIDIA L4 GPU (24GB VRAM)** with native **Vietnamese & English multilingual support** and rigorous benchmarking based on **Vectara Open RAG Bench** (`vectara/open-rag-bench`).

---

## Architectural Highlights

| Component | Selected Technology | Configuration & Benefits |
| :--- | :--- | :--- |
| **Compute Hardware** | Google Colab NVIDIA L4 GPU | 24GB GDDR6 VRAM, Ada Lovelace (CC 8.9), TF32 & SDPA hardware acceleration |
| **Multilingual LLM** | `Qwen/Qwen2.5-7B-Instruct` | **Native Bfloat16** precision (~14.5 GB VRAM) with fluent, natural Vietnamese comprehension |
| **Multilingual Embeddings**| `BAAI/bge-m3` | 1024-dimensional dense vectors, 8192-token context window, SOTA for Vietnamese |
| **Vector Store** | `ChromaDB` | Embedded vector database with metadata filtering and Drive persistence |
| **Document Parsing** | `PyMuPDF` (`fitz`) / `pypdf` | Multi-page PDF text & metadata extraction |
| **Evaluation Benchmark** | **Vectara Open RAG Bench** | Evaluates Hit Rate @ K, MRR, Token F1, ROUGE-L, Grounding, and Latency before Web UI |
| **User Interface** | Inline Chat + Gradio Web UI | Bilingual Gradio interface with `share=True` public link |

---

## Benchmark Methodology (Vectara Open RAG Bench)

Inserted immediately **before** the Interactive Web UI step:
1. **Retrieval Metrics**:
   - **Hit Rate @ K**: Verifies whether the ground truth context was retrieved within top-$k$ results.
   - **Mean Reciprocal Rank (MRR)**: Measures the reciprocal of the rank at which the first relevant document was retrieved.
2. **Generation Metrics**:
   - **Token F1 Score**: Measures word-level precision and recall against reference answers.
   - **ROUGE-L Score**: Measures longest common subsequence overlap.
   - **Hallucination Guardrail Check**: Tests out-of-domain queries to verify the model refuses to hallucinate.
   - **Query Latency**: Reports inference latency in milliseconds on NVIDIA L4 GPU.
3. **Dataset Source**:
   - Streaming support for Hugging Face `vectara/open_ragbench` (arXiv multimodal PDF benchmark) with curated bilingual fallback test cases.

---

## Notebook Structure (25 Cells)

- **Bước 1**: Hardware diagnostic (L4 GPU, 24GB VRAM, BF16, TF32)
- **Bước 2**: Dependency installation (`transformers`, `accelerate`, `bitsandbytes`, `sentence-transformers`, `chromadb`, `datasets`, `gradio`)
- **Bước 3**: Document ingestion (PDF upload or Vietnamese sample PDF generator)
- **Bước 4**: Multilingual text chunking (`chunk_size=800`, `chunk_overlap=160`)
- **Bước 5**: `BAAI/bge-m3` embedding model & ChromaDB indexing
- **Bước 6**: Semantic retrieval testing in Vietnamese
- **Bước 7**: `Qwen 2.5 7B` loaded in native Bfloat16 with SDPA
- **Bước 8**: RAG chain with anti-hallucination prompt & citations
- **Bước 9**: Inline interactive question answering
- **Bước 10 [BENCHMARK]**: Vectara Open RAG Benchmark evaluation runner & summary card
- **Bước 11 [INTERACTIVE WEB]**: Gradio bilingual web UI with public shareable link
- **Bước 12**: Memory garbage collection & Google Drive backup
