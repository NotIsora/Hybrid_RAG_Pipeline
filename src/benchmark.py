"""
Vectara Open RAG Benchmark Official Evaluation Engine
====================================================
Evaluates RAG systems on the full 3,045 QA pairs from the official
'vectara/open_ragbench' dataset (BEIR format with ArXiv multimodal papers).

Official Dataset Statistics:
- Total QA Pairs: 3,045 valid question-answer pairs
- By Query Types:
  * 1,793 abstractive queries (synthesis, rephrased response)
  * 1,252 extractive queries (concise, fact-based answers)
- By Generation Sources:
  * 1,914 text-only queries
  * 763 text-image queries
  * 148 text-table queries
  * 220 text-table-image queries
"""

import os
import json
import time
import urllib.request
import re
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict, Counter
import pandas as pd

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:
        def __init__(self, page_content: str, metadata: dict = None):
            self.page_content = page_content
            self.metadata = metadata or {}


HF_RAW_BASE = "https://huggingface.co/datasets/vectara/open_ragbench/raw/main/pdf/arxiv"


class VectaraOpenRAGBench:
    """Official benchmark runner for Vectara Open RAG Benchmark (3,045 QA pairs)."""

    def __init__(self, cache_dir: str = "./vectara_data"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(os.path.join(self.cache_dir, "corpus"), exist_ok=True)
        self.queries: Dict[str, Dict[str, Any]] = {}
        self.answers: Dict[str, str] = {}
        self.qrels: Dict[str, Dict[str, Any]] = {}
        self.pdf_urls: Dict[str, str] = {}

    def _fetch_json(self, relative_path: str) -> Any:
        local_path = os.path.join(self.cache_dir, relative_path.replace("/", os.sep))
        if os.path.exists(local_path):
            with open(local_path, "r", encoding="utf-8") as f:
                return json.load(f)

        url = f"{HF_RAW_BASE}/{relative_path}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=35) as resp:
            content = resp.read().decode("utf-8")

        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(content)

        return json.loads(content)

    def load_metadata(self) -> Tuple[int, int, int]:
        """Loads queries.json, answers.json, qrels.json, and pdf_urls.json."""
        self.queries = self._fetch_json("queries.json")
        self.answers = self._fetch_json("answers.json")
        self.qrels = self._fetch_json("qrels.json")
        try:
            self.pdf_urls = self._fetch_json("pdf_urls.json")
        except Exception:
            self.pdf_urls = {}
        return len(self.queries), len(self.answers), len(self.qrels)

    def print_dataset_statistics(self) -> Dict[str, Any]:
        """Validates and prints official dataset distribution."""
        if not self.queries:
            self.load_metadata()

        type_counts = defaultdict(int)
        source_counts = defaultdict(int)

        for q_id, q_info in self.queries.items():
            type_counts[q_info.get("type", "unknown")] += 1
            source_counts[q_info.get("source", "unknown")] += 1

        print("=" * 75)
        print("📊 VECTARA OPEN RAG BENCHMARK - OFFICIAL DATASET STRUCTURE")
        print(f"Total QA Pairs: {len(self.queries):,} valid question-answer pairs")
        print("\nBased on Query Types:")
        print(f"  • Abstractive: {type_counts.get('abstractive', 0):,} queries (summaries / synthesis)")
        print(f"  • Extractive:  {type_counts.get('extractive', 0):,} queries (concise, fact-based)")
        print("\nBased on Generation Sources:")
        print(f"  • Text-only:          {source_counts.get('text', 0):,} queries")
        print(f"  • Text-Image:         {source_counts.get('text-image', 0):,} queries")
        print(f"  • Text-Table:         {source_counts.get('text-table', 0):,} queries")
        print(f"  • Text-Table-Image:   {source_counts.get('text-table-image', 0):,} queries")
        print("=" * 75)

        return {
            "total_queries": len(self.queries),
            "type_counts": dict(type_counts),
            "source_counts": dict(source_counts)
        }

    def get_all_qa_pairs(self) -> List[Dict[str, Any]]:
        """Returns all 3,045 official question-answer pairs dynamically without hard-coding."""
        if not self.queries or not self.qrels or not self.answers:
            self.load_metadata()

        all_cases = []
        for q_id, q_rel in self.qrels.items():
            if q_id in self.queries and q_id in self.answers:
                q_meta = self.queries[q_id]
                all_cases.append({
                    "query_id": q_id,
                    "query": q_meta["query"],
                    "query_type": q_meta.get("type", "unknown"),
                    "source_modality": q_meta.get("source", "text"),
                    "golden_doc_id": q_rel["doc_id"],
                    "golden_section_id": q_rel.get("section_id"),
                    "ground_truth": self.answers[q_id]
                })
        return all_cases

    def get_sampled_qa_pairs(
        self,
        custom_limits: Optional[Dict[str, Optional[int]]] = None,
        filter_indexed_docs: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """
        Samples questions directly based solely on custom_limits dictionary:
        - custom_limits: dict specifying count per category (e.g. {'abstractive': 10, 'extractive': 10})
        - filter_indexed_docs: set of doc_ids present in the vector store to prioritize/filter
        """
        if custom_limits is None:
            custom_limits = {"abstractive": 5, "extractive": 5}

        all_cases = self.get_all_qa_pairs()

        # Prioritize queries whose target documents are actually indexed
        if filter_indexed_docs:
            doc_set = set(filter_indexed_docs)
            indexed_cases = [c for c in all_cases if c["golden_doc_id"] in doc_set]
            other_cases = [c for c in all_cases if c["golden_doc_id"] not in doc_set]
            all_cases = indexed_cases + other_cases

        sampled_cases = []
        seen_ids = set()

        for category, limit in custom_limits.items():
            matching = [
                c for c in all_cases
                if (c.get("query_type") == category or c.get("source_modality") == category)
                and c["query_id"] not in seen_ids
            ]
            selected = matching if (limit is None or limit < 0) else matching[:limit]
            for item in selected:
                seen_ids.add(item["query_id"])
                sampled_cases.append(item)

        return sampled_cases

    def fetch_corpus_paper(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Downloads structured JSON paper from official corpus."""
        filename = f"corpus/{doc_id}.json"
        try:
            return self._fetch_json(filename)
        except Exception as e:
            return None

    def prepare_corpus_documents(self, max_papers: Optional[int] = None) -> List[Document]:
        """
        Converts arXiv papers referenced in qrels into searchable LangChain Document objects.
        """
        if not self.qrels:
            self.load_metadata()

        unique_docs = list(dict.fromkeys(q["doc_id"] for q in self.qrels.values()))
        if max_papers is not None:
            unique_docs = unique_docs[:max_papers]

        documents = []
        for doc_id in unique_docs:
            paper_data = self.fetch_corpus_paper(doc_id)
            if not paper_data:
                continue

            title = paper_data.get("title", f"arXiv:{doc_id}").strip()
            sections = paper_data.get("sections", [])
            for sec_idx, sec in enumerate(sections):
                text_content = sec.get("text", "").strip()
                tables = sec.get("tables", {})
                if tables:
                    text_content += "\n\nTables:\n" + "\n".join(tables.values())

                if len(text_content) < 30:
                    continue

                doc = Document(
                    page_content=text_content,
                    metadata={
                        "doc_id": doc_id,
                        "section_id": sec_idx,
                        "title": title,
                        "filename": f"{doc_id}.pdf"
                    }
                )
                documents.append(doc)

        return documents

    @staticmethod
    def _normalize_answer(s: str) -> str:
        """Lowercases, removes punctuation and English articles according to standard SQuAD/BEIR evaluation."""
        s = s.lower().strip()
        s = re.sub(r"\b(a|an|the)\b", " ", s)
        s = re.sub(r"[^\w\s]", " ", s)
        return " ".join(s.split())

    @classmethod
    def compute_token_f1(cls, pred: str, gt: str) -> float:
        pred_clean = cls._normalize_answer(pred).split()
        gt_clean = cls._normalize_answer(gt).split()
        if not pred_clean or not gt_clean:
            return 1.0 if pred_clean == gt_clean else 0.0
        common = Counter(pred_clean) & Counter(gt_clean)
        num_same = sum(common.values())
        if num_same == 0:
            return 0.0
        p = 1.0 * num_same / len(pred_clean)
        r = 1.0 * num_same / len(gt_clean)
        return 2.0 * p * r / (p + r)

    @classmethod
    def compute_rouge_l(cls, pred: str, gt: str) -> float:
        p_tokens = cls._normalize_answer(pred).split()
        g_tokens = cls._normalize_answer(gt).split()
        m, n = len(p_tokens), len(g_tokens)
        if m == 0 or n == 0:
            return 0.0
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if p_tokens[i - 1] == g_tokens[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        lcs = dp[m][n]
        p = lcs / m
        r = lcs / n
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

    @classmethod
    def compute_semantic_f1(cls, pred: str, gt: str, embedding_model=None) -> float:
        """Computes semantic similarity / Semantic F1 using dense embeddings (e.g. bge-m3)."""
        if not pred.strip() or not gt.strip():
            return 1.0 if pred.strip() == gt.strip() else 0.0
        if embedding_model is not None:
            import numpy as np
            vecs = embedding_model.embed_documents([pred, gt])
            v1, v2 = np.array(vecs[0], dtype=np.float32), np.array(vecs[1], dtype=np.float32)
            norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return float(max(0.0, min(1.0, np.dot(v1, v2) / (norm1 * norm2))))
        return cls.compute_token_f1(pred, gt)

    def evaluate_retrieval_all(self, retriever_func, top_k_list: List[int] = [1, 4, 10]) -> pd.DataFrame:
        """
        Executes full retrieval benchmark across ALL 3,045 queries.
        Computes Hit@K and MRR for each query type and modality source.
        """
        qa_pairs = self.get_all_qa_pairs()
        print(f"\n🚀 Evaluating Retrieval on ALL {len(qa_pairs):,} Vectara Open RAG Bench QA pairs...")

        max_k = max(top_k_list)
        results = []

        for idx, item in enumerate(qa_pairs, 1):
            q = item["query"]
            gold_doc = item["golden_doc_id"]

            t0 = time.time()
            retrieved_docs = retriever_func(q, max_k)
            lat_ms = (time.time() - t0) * 1000

            doc_hits = {}
            for k in top_k_list:
                doc_hits[f"Hit@{k}"] = 0

            mrr_val = 0.0
            for rank, doc in enumerate(retrieved_docs, 1):
                doc_meta = doc.metadata if hasattr(doc, "metadata") else doc
                doc_fn = str(doc_meta.get("filename", "") or doc_meta.get("doc_id", ""))
                if gold_doc in doc_fn:
                    if mrr_val == 0.0:
                        mrr_val = 1.0 / rank
                    for k in top_k_list:
                        if rank <= k:
                            doc_hits[f"Hit@{k}"] = 1
                    break

            row = {
                "query_id": item["query_id"],
                "query_type": item["query_type"],
                "source_modality": item["source_modality"],
                "MRR": mrr_val,
                "latency_ms": lat_ms
            }
            row.update(doc_hits)
            results.append(row)

        df = pd.DataFrame(results)

        # Aggregate overall and by categories
        summary_rows = []

        def get_summary_row(sub_df, category_name):
            n = len(sub_df)
            if n == 0:
                return {}
            s = {"Category": category_name, "Count": n}
            for k in top_k_list:
                s[f"Hit@{k} (%)"] = round(sub_df[f"Hit@{k}"].mean() * 100, 2)
            s["MRR"] = round(sub_df["MRR"].mean(), 4)
            s["Avg Latency (ms)"] = round(sub_df["latency_ms"].mean(), 2)
            return s

        summary_rows.append(get_summary_row(df, "OVERALL (All QA Pairs)"))
        for q_type in ["abstractive", "extractive"]:
            summary_rows.append(get_summary_row(df[df["query_type"] == q_type], f"Type: {q_type}"))
        for s_mod in ["text", "text-image", "text-table", "text-table-image"]:
            summary_rows.append(get_summary_row(df[df["source_modality"] == s_mod], f"Source: {s_mod}"))

        summary_df = pd.DataFrame(summary_rows)
        return summary_df

    def evaluate_e2e_benchmark(
        self,
        rag_query_func,
        eval_cases: List[Dict[str, Any]],
        top_k: int = 4
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs end-to-end evaluation (Retrieval Hit@K + MRR + Generation Token F1 + ROUGE-L + Latency).
        Returns:
            - details_df: Per-query evaluation metrics
            - summary_df: Aggregate metrics grouped by query_type and source_modality
        """
        results = []
        print(f"\n🚀 Running End-to-End Benchmark on {len(eval_cases)} QA pairs (Top-K={top_k})...")

        for idx, item in enumerate(eval_cases, 1):
            q = item["query"]
            gold = item["ground_truth"]
            gold_doc = item["golden_doc_id"]
            q_type = item.get("query_type", "unknown")
            s_mod = item.get("source_modality", "unknown")

            t0 = time.time()
            res = rag_query_func(q, top_k=top_k)
            latency_ms = (time.time() - t0) * 1000

            pred = res.get("answer", "")
            sources = res.get("sources", [])

            # Retrieval check
            hit_at_k = 0
            mrr_val = 0.0
            for rank, src in enumerate(sources, 1):
                src_meta = src.get("metadata", {}) if isinstance(src, dict) else (src.metadata if hasattr(src, "metadata") else {})
                src_id = str(src_meta.get("filename", "") or src_meta.get("doc_id", ""))
                if gold_doc in src_id:
                    if mrr_val == 0.0:
                        mrr_val = 1.0 / rank
                    if rank <= top_k:
                        hit_at_k = 1
                    break

            f1 = self.compute_token_f1(pred, gold)
            rouge_l = self.compute_rouge_l(pred, gold)

            results.append({
                "ID": idx,
                "Query": q[:45] + ("..." if len(q) > 45 else ""),
                "Type": q_type,
                "Source": s_mod,
                "Hit@K": hit_at_k,
                "MRR": round(mrr_val, 3),
                "Token F1": round(f1, 3),
                "ROUGE-L": round(rouge_l, 3),
                "Latency (ms)": round(latency_ms, 1)
            })

            print(f"[{idx:02d}/{len(eval_cases):02d}] Hit: {hit_at_k} | F1: {f1:.3f} | ROUGE-L: {rouge_l:.3f} | {latency_ms:.0f}ms | [{q_type}] {q[:35]}...")

        details_df = pd.DataFrame(results)

        # Summary by category
        summary_rows = []
        def get_summary(sub_df, name):
            if len(sub_df) == 0:
                return {}
            return {
                "Category": name,
                "Count": len(sub_df),
                f"Hit@{top_k} (%)": round(sub_df["Hit@K"].mean() * 100, 1),
                "MRR": round(sub_df["MRR"].mean(), 3),
                "Token F1": round(sub_df["Token F1"].mean(), 3),
                "ROUGE-L": round(sub_df["ROUGE-L"].mean(), 3),
                "Avg Latency (ms)": round(sub_df["Latency (ms)"].mean(), 1)
            }

        summary_rows.append(get_summary(details_df, "OVERALL"))
        for t in details_df["Type"].unique():
            summary_rows.append(get_summary(details_df[details_df["Type"] == t], f"Type: {t}"))
        for s in details_df["Source"].unique():
            summary_rows.append(get_summary(details_df[details_df["Source"] == s], f"Source: {s}"))

        summary_df = pd.DataFrame(summary_rows)
        return details_df, summary_df
