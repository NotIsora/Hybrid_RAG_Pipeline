<div align="center">

# 🔬 High-Performance Scientific RAG Pipeline
### Multi-Stage Hybrid Retrieval & Cross-Encoder Reranking for Research Literature

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.4+](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![NVIDIA A100](https://img.shields.io/badge/Hardware-NVIDIA%20A100%20(80GB)-76B900.svg?logo=nvidia&logoColor=white)](https://www.nvidia.com/en-us/data-center/a100/)
[![HuggingFace Models](https://img.shields.io/badge/Models-Qwen%202.5%2014B%20%7C%20BGE--M3-FFD21E.svg?logo=huggingface&logoColor=black)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

**A production-grade, end-to-end Retrieval-Augmented Generation (RAG) system specialized for scientific research literature.**  
Engineered for **NVIDIA A100 (40GB/80GB)** and **L4 (24GB)** Tensor Core GPUs, validated across all **3,045 questions** on the official **[Vectara Open RAG Bench](https://github.com/vectara/open-rag-bench)**.

[Overview](#-overview) • [Key Strategies](#-key-accuracy-strategies) • [Benchmark Results](#-benchmark-results--visual-analytics) • [Architecture](#-system-architecture) • [Quickstart](#-quickstart--google-colab) • [Web UI](#-interactive-web-interface)

---

</div>

## 🌟 Overview

Analyzing scientific research requires understanding dense mathematical formulations, complex chemical/biological terminology, structured tabular data, and multimodal figures. Standard naive RAG pipelines often suffer from low precision, lost tabular context, and hallucinations.

This pipeline introduces an **industrial-strength, multi-stage architecture** designed to solve these exact challenges:
* **Near-Perfect Retrieval**: Achieves **95.3% Hit@1** and **99.3% Hit@6** across 396 full arXiv papers on the official Vectara benchmark.
* **High-Fidelity Generation**: Synthesizes factual, concise responses powered by **`Qwen/Qwen2.5-14B-Instruct`** in native **Bfloat16** with **SDPA FlashAttention**.
* **Sub-Second Throughput**: Delivers an end-to-end latency of **867.5 ms per query** on an NVIDIA A100 GPU (batch size = 8).
* **Obsidian Dark UI**: A publication-grade web interface featuring Google Antigravity's dark aesthetic (`#0D0E12`), real-time citation tracking, and dynamic PDF ingestion.

---

## 🎯 Key Accuracy Strategies

The pipeline incorporates four advanced strategies engineered specifically for scientific papers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 1. Dual-Stage Hybrid Retrieval                                             │
│    Dense BAAI/bge-m3 (1024d) + Sparse BM25 Okapi fused via RRF (k=60)       │
│    └─► Combines deep semantic nuance with exact formula/acronym matching.   │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 2. Native BF16 Cross-Encoder Re-Ranking                                     │
│    BAAI/bge-reranker-large (560M params) evaluates full token cross-attention│
│    └─► Elevates Hit@1 from 92.0% to 95.3% and Hit@6 to 99.3%.               │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 3. Structured Table Ingestion & Section Breadcrumbs                         │
│    Preserves table grids with [TABLE]...[/TABLE] & injects document tags     │
│    └─► Guarantees spatial and contextual integrity across tables & figures. │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ 4. Question-Anchored Declarative Synthesis (Echo Framing)                  │
│    Direct 1-2 sentence declarative statements with exact factual grounding  │
│    └─► Aligns with gold scientific standards; Semantic F1: 0.820 (abstractive)│
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Hybrid Retrieval (Dense BGE-M3 + Sparse BM25 Okapi)**:
   - Dense embeddings capture conceptual and thematic context across an 8,192-token context window.
   - Lexical BM25 ensures exact-match fidelity for alphanumeric identifiers, gene names, chemical symbols, and numeric constants.
   - Fused via **Reciprocal Rank Fusion (RRF)**:

     $$\text{RRF}(d) = \sum_{m \in \{\text{dense},\, \text{sparse}\}} \frac{1}{k + \text{rank}_m(d)} \quad (k = 60)$$
2. **Cross-Encoder Re-Ranking (`BAAI/bge-reranker-large`)**:
   - Re-evaluates top candidate pairs $(q, p)$ via deep multi-head cross-attention in native Bfloat16.
   - Selects the top-4 most confident passages for generation context, filtering out lower-ranked distractors.
3. **Structured Table & Section Breadcrumbs**:
   - Extracts and encapsulates tables inside `[TABLE id='...']\n...\n[/TABLE]` markdown blocks.
   - Context is injected with document and section coordinates: `[Document {i}: {title} | Section {sec}]`.
4. **Question-Anchored Declarative Synthesis**:
   - Formulates responses in direct, well-formed declarative sentences echoing the question's core subject.
   - Eliminates conversational preambles (no *"Based on the text..."*), maximizing factual precision and citation clarity.

---

## 🏆 Benchmark Results & Visual Analytics

Evaluated across all **3,045 official queries** on the full **[Vectara Open RAG Bench](https://github.com/vectara/open-rag-bench)** (396 arXiv papers) on an **NVIDIA A100 GPU (80GB VRAM)**:

| Category | Query Count | Hit@1 (%) | Hit@6 (%) | MRR | Token F1 | Semantic F1 | ROUGE-L | Avg Latency (ms) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **OVERALL** | **3,045** | **95.3%** | **99.3%** | **0.970** | **0.416** | **0.784** | **0.367** | **867.5 ms** |
| **Type: abstractive** | 1,793 | 94.9% | 99.2% | 0.966 | 0.385 | **0.820** | 0.318 | 873.9 ms |
| **Type: extractive** | 1,252 | 96.0% | 99.4% | 0.975 | **0.460** | 0.733 | **0.437** | 858.3 ms |
| **Source: text** | 1,914 | **96.7%** | **99.7%** | **0.979** | 0.455 | 0.786 | 0.412 | 866.8 ms |
| **Source: text-table** | 148 | 95.9% | 99.3% | 0.974 | 0.374 | 0.776 | 0.310 | 857.8 ms |
| **Source: text-image** | 763 | 92.4% | 98.4% | 0.949 | 0.347 | 0.786 | 0.287 | 874.2 ms |
| **Source: text-table-image** | 220 | 93.6% | 98.6% | 0.957 | 0.345 | 0.768 | 0.288 | 856.4 ms |

### Key Benchmark Insights:
* **State-of-the-Art Retrieval**: Overall **Hit@6 reached 99.3%** and **Hit@1 reached 95.3%** (MRR = 0.970), ensuring the ground-truth evidence is nearly always the top-ranked passage.
* **Complex Multimodal Resilience**: Even on compound scientific queries combining text, tables, and images (`text-table-image`), Hit@1 achieved **93.6%** and Hit@6 achieved **98.6%**.
* **High Semantic Alignment**: Abstractive queries achieved a **Semantic F1 of 0.820**, demonstrating deep contextual fidelity to peer-reviewed scientific findings.

---

### 📊 Publication-Grade Benchmark Visualizations

The pipeline generates dedicated, high-resolution individual analytics charts:

#### Figure 1: Retrieval Accuracy (Hit@1 vs. Hit@6 across all subsets)
![Figure 1: Retrieval Accuracy](assets/retrieval.png)

#### Figure 2: Generation Quality Metrics (Semantic F1, Token F1, ROUGE-L)
![Figure 2: Generation Quality](assets/generation.png)

#### Figure 3: Modality Source Breakdown (Text vs. Tables vs. Figures)
![Figure 3: Modality Breakdown](assets/modality.png)

#### Figure 4: End-to-End Latency Profile per Query Category
![Figure 4: Latency Profile](assets/latency.png)

---

## 🏛️ System Architecture

```mermaid
flowchart TD
    subgraph INGESTION ["📥 Document Ingestion & Indexing"]
        A["ArXiv PDF / JSON Corpus"] --> B["Section & Table Extractor"]
        B --> C["Structured Chunks<br/><code>[TABLE]...[/TABLE]</code>"]
        C --> D["Dense Vector Store<br/><b>ChromaDB (BAAI/bge-m3)</b>"]
        C --> E["Sparse Lexical Index<br/><b>BM25 Okapi</b>"]
    end

    subgraph RETRIEVAL ["🔍 Two-Stage Hybrid Retrieval"]
        Q["User Research Query"] --> F["Dense Top-12 Search"]
        Q --> G["Sparse Top-12 Search"]
        D --> F
        E --> G
        F --> H["Reciprocal Rank Fusion<br/><b>RRF (k=60)</b>"]
        G --> H
        H --> I["Top-12 Hybrid Candidates"]
        I --> J["Cross-Encoder Re-Ranking<br/><b>BAAI/bge-reranker-large (BF16)</b>"]
        Q --> J
        J --> K["Top-4 High-Confidence Passages"]
    end

    subgraph GENERATION ["🧠 Factual Grounded Generation"]
        K --> L["Context Injection + Breadcrumbs"]
        Q --> M["Chat Template + System Prompt"]
        L --> M
        M --> N["Qwen 2.5 14B Instruct<br/><b>Native BF16 + SDPA</b>"]
        N --> O["Precision Factual Answer + Citations"]
    end

    subgraph UI ["🌐 Antigravity Dark Interface"]
        O --> P["Interactive Dialogue & Citation Cards"]
    end
```

---

## ⚡ Hardware Optimizations

| Optimization | Implementation Details | Impact |
|---|---|---|
| **Native Bfloat16** | Unquantized FP16/BF16 tensor execution on Ampere Tensor Cores | Eliminates quantization drift; preserves decimal accuracy |
| **SDPA FlashAttention** | PyTorch native `attn_implementation="sdpa"` | Memory footprint reduction & $2.4\times$ faster prefill |
| **Batch Inference** | Left-padded tokenizer with `batch_size = 8` on A100 | Slashes per-query generation latency to ~867 ms |
| **VRAM Management** | `PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"` | Prevents CUDA memory fragmentation during long queries |

---

## 🚀 Quickstart & Google Colab

### Option 1: Run on Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com).
2. Go to **File** ➔ **Upload notebook**, and upload [`notebooks/rag_colab.ipynb`](notebooks/rag_colab.ipynb).
3. Navigate to **Runtime** ➔ **Change runtime type** ➔ select **A100 GPU** (or L4 GPU).
4. Run all cells:
   - **Section 1**: Initializes models, downloads papers, and builds indexes.
   - **Section 2**: Executes the Vectara benchmark and renders the analytics charts.
   - **Section 3**: Launches the interactive Gradio web application with a public share link (`share=True`).

### Option 2: Local Installation

```bash
# 1. Clone repository
git clone https://github.com/NotIsora/RAG.git
cd RAG

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Option 3: Python Programmatic Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import torch

# 1. Query the RAG Pipeline
query = "What is the computational complexity of sparse dot-product attention?"

# Example response object
result = rag_query(query)
print("Answer:", result["answer"])
print("Top Citation:", result["sources"][0]["title"])
```

---

## ⚙️ Benchmark Configuration: `CUSTOM_LIMITS`

In **Section 2** of the pipeline, you can adjust the benchmark evaluation scope using `CUSTOM_LIMITS`:

```python
# Evaluate 100% of all 3,045 queries (Full Evaluation):
CUSTOM_LIMITS = None

# Or sample specific subsets for rapid testing:
CUSTOM_LIMITS = {
    "abstractive": 50,
    "extractive": 50,
    "text": 40,
    "text-table": 20,
    "text-image": 20,
    "text-table-image": 20
}
```

---

## 🌐 Interactive Web Interface

The pipeline includes an interactive web interface built with **Gradio 6.0+**, featuring Google Antigravity's signature **Obsidian Dark Aesthetic**:

* **Color Palette**: Deep Obsidian background (`#0D0E12`), card panels (`#15171E`), hairline borders (`#232631`), and Indigo accents (`#4F46E5`).
* **Zero Jitter**: Completely eliminated scale transforms on hover for a calm, professional reading experience.
* **Citations Drawer**: Structured citation cards displaying the paper title, filename, section number, and exact verbatim context snippet.
* **Dynamic PDF Ingestion**: Drag-and-drop any research paper to automatically parse and index it into both ChromaDB and BM25 on the fly.
* **KPI Metric Tracker**: Live scorecard showing retrieval accuracy, MRR, and Semantic F1.

---

## 📁 Repository Structure

```
├── assets/
│   ├── retrieval.png             # Figure 1: Hit@1 & Hit@6 chart (300 DPI)
│   ├── generation.png            # Figure 2: Semantic F1, Token F1, ROUGE-L chart
│   ├── modality.png              # Figure 3: Multimodal source comparison chart
│   ├── latency.png               # Figure 4: End-to-end latency distribution chart
│   └── overview.png              # Composite 4-panel dashboard
├── data/
│   ├── paper1.pdf                # Sample research paper 1 (arXiv:2410.14077)
│   ├── paper2.pdf                # Sample research paper 2 (arXiv:2401.07294)
│   └── sample.txt                # Reference text
├── notebooks/
│   ├── rag_colab.ipynb           # Complete 11-cell Google Colab notebook (A100/L4 GPU optimized)
│   └── build.py                  # Notebook builder script
├── src/
│   ├── rag.py                    # Core RAG engine with BGE-M3, BM25, and Qwen 2.5 14B
│   ├── benchmark.py              # Vectara Open RAG Bench dataset loader & harness
│   ├── metrics.py                # BEIR / SQuAD F1, Semantic F1, and ROUGE-L metrics
│   └── gpu.py                    # Hardware diagnostics (Compute capability, TF32, VRAM)
├── .gitignore                    # Git ignore file
├── LICENSE                       # MIT License
├── requirements.txt              # Locked dependencies
└── README.md                     # Documentation
```

---

## 📜 Citations & Acknowledgments

* **Vectara Open RAG Bench**: [arXiv:2410.14077](https://arxiv.org/abs/2410.14077) – *Open RAG Bench: Benchmarking Retrieval-Augmented Generation for Complex Domain Tasks*.
* **Qwen 2.5 Team**: Alibaba Cloud for releasing `Qwen/Qwen2.5-14B-Instruct`.
* **BAAI**: Beijing Academy of Artificial Intelligence for `BAAI/bge-m3` and `BAAI/bge-reranker-large`.
* **ChromaDB & LangChain**: Open-source infrastructure powering dense vector search.

---

<div align="center">
Developed for High-Precision Scientific AI Research • Licensed under the MIT License.
</div>
