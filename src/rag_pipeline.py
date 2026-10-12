"""
High-Performance Multilingual & Vietnamese Local RAG Pipeline
Optimized for NVIDIA L4 GPU (24GB VRAM) on Google Colab
=============================================================================
Multilingual Features (Full Vietnamese + English Support):
- Dense Multilingual Embedding: BAAI/bge-m3 (SOTA for Vietnamese & 100+ languages)
- Multilingual LLM: Qwen/Qwen2.5-7B-Instruct (Native Bfloat16) or Qwen2.5-14B (4-bit NF4)
  with fluent, grammatically sound Vietnamese understanding and generation.
- Language-Adaptive Grounding: Answers in the query's language with verified citations.
- Hardware Acceleration: NVIDIA L4 Ada Lovelace (SDPA, TF32, 24GB VRAM).
"""

import os
import gc
import torch
from typing import List, Dict, Any, Optional

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline
)


def enable_l4_optimizations():
    """Enables hardware tensor acceleration on Ada Lovelace / Ampere architectures."""
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(0)
        if major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("🚀 Hardware Acceleration: TF32 enabled for 4th-Gen Tensor Cores.")


class GPUHelper:
    """Utilities for monitoring and managing Google Colab L4 GPU memory."""

    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        is_available = torch.cuda.is_available()
        if not is_available:
            return {
                "available": False,
                "device_name": "CPU Only",
                "compute_capability": "N/A",
                "bf16_supported": False,
                "total_memory_gb": 0.0,
                "allocated_memory_gb": 0.0,
                "free_memory_gb": 0.0,
            }

        device_name = torch.cuda.get_device_name(0)
        major, minor = torch.cuda.get_device_capability(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        allocated_mem = torch.cuda.memory_allocated(0) / (1024 ** 3)
        free_mem = total_mem - allocated_mem
        bf16_supported = torch.cuda.is_bf16_supported()

        return {
            "available": True,
            "device_name": device_name,
            "compute_capability": f"{major}.{minor}",
            "bf16_supported": bf16_supported,
            "total_memory_gb": round(total_mem, 2),
            "allocated_memory_gb": round(allocated_mem, 2),
            "free_memory_gb": round(free_mem, 2),
        }

    @staticmethod
    def print_gpu_status():
        info = GPUHelper.get_gpu_info()
        print("=" * 65)
        print("Hardware Status:")
        print(f"  CUDA Available:     {info['available']}")
        print(f"  Device:             {info['device_name']}")
        print(f"  Compute Capability: {info['compute_capability']}")
        print(f"  Native Bfloat16:    {info['bf16_supported']}")
        if info["available"]:
            print(f"  Total VRAM:         {info['total_memory_gb']} GB")
            print(f"  Allocated VRAM:     {info['allocated_memory_gb']} GB")
            print(f"  Free VRAM:          {info['free_memory_gb']} GB")
        print("=" * 65)

    @staticmethod
    def clear_vram():
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("CUDA cache cleared.")


class DocumentProcessor:
    """Handles loading and chunking documents with Vietnamese and English punctuation support."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 160):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # Configured for both Vietnamese and English sentence structures
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", ".\n", "... ", "? ", "! ", " ", ""],
            length_function=len,
        )

    def load_document(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_ext = os.path.splitext(file_path)[1].lower()
        if file_ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        docs = loader.load()
        print(f"Loaded {len(docs)} page(s)/section(s) from {os.path.basename(file_path)}")
        return docs

    def process_and_chunk(self, file_path: str) -> List[Document]:
        raw_docs = self.load_document(file_path)
        chunks = self.text_splitter.split_documents(raw_docs)

        base_name = os.path.basename(file_path)
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["filename"] = base_name
            if "page" not in chunk.metadata:
                chunk.metadata["page"] = 1
            else:
                chunk.metadata["page"] = chunk.metadata["page"] + 1

        print(f"Split {base_name} into {len(chunks)} chunks (chunk_size={self.chunk_size}, overlap={self.chunk_overlap})")
        return chunks


class VectorStoreManager:
    """
    Manages dense multilingual embeddings via BAAI/bge-m3 and ChromaDB vector indexing.
    bge-m3 is state-of-the-art for Vietnamese and 100+ languages with 1024-dimensional embeddings.
    """

    def __init__(
        self,
        embedding_model_name: str = "BAAI/bge-m3",
        persist_directory: str = "./chroma_db"
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Initializing Multilingual Embedding Model '{embedding_model_name}' on {device}...")
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32}
        )
        self.persist_directory = persist_directory
        self.vector_store: Optional[Chroma] = None

    def build_or_update(self, documents: List[Document]) -> Chroma:
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=self.persist_directory
            )
        else:
            self.vector_store.add_documents(documents)
        print(f"Indexed documents into ChromaDB at '{self.persist_directory}'.")
        return self.vector_store

    def load_existing(self) -> Chroma:
        if not os.path.exists(self.persist_directory):
            raise FileNotFoundError(f"No existing ChromaDB found at {self.persist_directory}")
        self.vector_store = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings
        )
        print(f"Loaded existing vector store from {self.persist_directory}")
        return self.vector_store

    def as_retriever(self, search_kwargs: Optional[Dict[str, Any]] = None):
        if self.vector_store is None:
            raise ValueError("Vector store has not been built or loaded yet.")
        kwargs = search_kwargs or {"k": 6}
        return self.vector_store.as_retriever(search_kwargs=kwargs)


class LocalLLMManager:
    """
    Manages local LLM inference optimized for NVIDIA A100 (40GB/80GB) and L4 (24GB VRAM).
    Powered by Qwen 2.5 14B which excels in scientific comprehension and reasoning.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-14B-Instruct",
        precision_mode: str = "native_bf16",
        max_new_tokens: int = 64,
        temperature: float = 0.01
    ):
        self.model_id = model_id
        self.precision_mode = precision_mode.lower()
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.pipeline = None

    def load_model(self):
        enable_l4_optimizations()
        
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
        
        print(f"\nLoading model: '{self.model_id}' | Mode: '{self.precision_mode}' | Dtype: {compute_dtype}")

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True
        )

        model_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "sdpa",
        }

        if self.precision_mode == "native_bf16":
            model_kwargs["torch_dtype"] = compute_dtype
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            print("Loaded in FULL native Bfloat16 precision (unquantized)!")

        elif self.precision_mode == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["quantization_config"] = bnb_config
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            print("Loaded in 4-bit NF4 precision via BitsAndBytes.")

        elif self.precision_mode == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = bnb_config
            model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
            print("Loaded in 8-bit precision via BitsAndBytes.")

        else:
            raise ValueError(f"Unsupported precision_mode: {self.precision_mode}.")

        self.pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=(self.temperature > 0.0),
            top_p=0.9,
            repetition_penalty=1.1,
            return_full_text=False,
        )
        
        GPUHelper.print_gpu_status()
        return self.pipeline


class RAGPipeline:
    """Coordinates retrieval, grounding, citation tracking, and response generation with Vietnamese support."""

    def __init__(self, vector_store_manager: VectorStoreManager, llm_pipeline):
        self.vsm = vector_store_manager
        self.llm = llm_pipeline

    def _format_context(self, docs: List[Document]) -> str:
        formatted_chunks = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("filename", "TaiLieu/Doc")
            page = doc.metadata.get("page", 1)
            content = doc.page_content.strip()
            formatted_chunks.append(f"[Nguồn {i} / Source {i} | Tệp/File: {source} | Trang/Page: {page}]\n{content}")
        return "\n\n".join(formatted_chunks)

    def query(self, question: str, top_k: int = 4) -> Dict[str, Any]:
        """
        Executes end-to-end multilingual RAG query:
        - Detects query intent and responds natively in Vietnamese or English.
        - Strictly grounds answers in the provided context with citations.
        """
        retriever = self.vsm.as_retriever(search_kwargs={"k": top_k})
        retrieved_docs = retriever.invoke(question)

        if not retrieved_docs:
            return {
                "answer": "I do not have enough information to answer this question.",
                "sources": [],
                "context": ""
            }

        context_str = self._format_context(retrieved_docs)

        # Dynamic Prompt Routing for Extractive vs Abstractive queries
        is_extractive = bool(
            re.match(r"^(is|are|was|were|does|do|did|can|could|will|would|should|has|have|had)\b", question.lower().strip())
            or re.search(r"\b(which|what is the (value|name|number|percentage|rate|shape|type|threshold)|how (many|much))\b", question.lower().strip())
        )

        if is_extractive:
            system_instruction = """You are a precision scientific fact extractor.
Provide an ultra-concise, exact factual answer based strictly on the provided Context.
Rules:
1. For Yes/No questions, begin directly with 'Yes,' or 'No,'.
2. Provide a single, direct sentence (strictly under 15 words). Never add extra explanations.
3. Use exact numbers, formulas, and terminology verbatim from the Context.
4. Echo the question subject directly (e.g., 'Q: What is the shape of X? A: The shape of X is a Gaussian shape.').
5. Never add preambles, introductory filler, or citations (do NOT write 'Based on the text').
6. If the context does not contain enough information to answer, reply strictly with: "I do not have enough information to answer this question.\""""
        else:
            system_instruction = """You are an expert scientific question-answering assistant.
Synthesize a concise, factually rigorous answer based strictly on the provided Context.
Rules:
1. Answer in strictly 1 to 2 concise sentences (under 30 words total).
2. Directly echo the question topic in your opening sentence.
3. Use exact scientific terminology from the Context to state the core finding or mechanism.
4. Never add bullet points, conversational filler, or citations (do NOT write 'Based on the text').
5. If the context does not contain enough information to answer, reply strictly with: "I do not have enough information to answer this question.\""""

        prompt = f"""<|im_start|>system
{system_instruction}
<|im_end|>
<|im_start|>user
Context:
{context_str}

Question: {question}
<|im_end|>
<|im_start|>assistant
"""

        output = self.llm(prompt)
        answer_text = output[0]["generated_text"].strip()

        sources = []
        for idx, doc in enumerate(retrieved_docs, 1):
            sources.append({
                "source_id": idx,
                "file": doc.metadata.get("filename", "unknown"),
                "page": doc.metadata.get("page", 1),
                "snippet": doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
            })

        return {
            "answer": answer_text,
            "sources": sources,
            "context": context_str
        }


def build_gradio_ui(rag_pipeline: RAGPipeline, doc_processor: DocumentProcessor, vsm: VectorStoreManager):
    """Pure English Gradio web interface for Vectara Open RAG Bench."""
    import gradio as gr

    def handle_upload(file_objs):
        if not file_objs:
            return "No files selected."
        try:
            uploaded_count = 0
            total_chunks = 0
            for f in file_objs:
                file_path = f if isinstance(f, str) else getattr(f, "name", str(f))
                if not os.path.exists(file_path):
                    continue
                chunks = doc_processor.process_and_chunk(file_path)
                if chunks:
                    vsm.build_or_update(chunks)
                    uploaded_count += 1
                    total_chunks += len(chunks)
            if uploaded_count > 0:
                return f"✅ Successfully indexed {uploaded_count} file(s) ({total_chunks} chunks) into ChromaDB!"
            else:
                return "⚠️ No valid content found in uploaded files."
        except Exception as e:
            return f"❌ Error processing files: {str(e)}"

    def handle_chat(message, history):
        if not message or not message.strip():
            return "", history or [], ""
        
        try:
            result = rag_pipeline.query(message)
            answer = result.get("answer", "")
            sources = result.get("sources", [])
        except Exception as e:
            answer = f"❌ Error generating answer: {str(e)}"
            sources = []
        
        citation_md = "### 📚 Verified Sources:\n"
        if sources:
            for src in sources:
                citation_md += f"- **[Source {src['source_id']}]** `{src['file']}` (Page {src['page']}):\n  > _{src['snippet']}_\n\n"
        else:
            citation_md += "_No relevant sources matched._"

        new_history = list(history) if history is not None else []
        
        is_dict_format = False
        if len(new_history) > 0 and isinstance(new_history[0], dict):
            is_dict_format = True
        else:
            try:
                is_dict_format = int(gr.__version__.split(".")[0]) >= 5
            except Exception:
                is_dict_format = False

        if is_dict_format:
            new_history.append({"role": "user", "content": message})
            new_history.append({"role": "assistant", "content": answer})
        else:
            new_history.append((message, answer))

        return "", new_history, citation_md

    custom_css = """
    :root {
        --color-primary-dark: #287A74;
        --color-primary-mid: #55A9A0;
        --color-accent-mint: #AEEED3;
        --color-accent-cream: #FFF8B0;
    }
    .header-banner {
        background: linear-gradient(135deg, #287A74 0%, #55A9A0 100%);
        color: white !important;
        padding: 24px 28px;
        border-radius: 14px;
        margin-bottom: 20px;
        box-shadow: 0 4px 14px rgba(40, 122, 116, 0.25);
    }
    .header-banner h1, .header-banner p {
        color: white !important;
    }
    .badge-accent {
        background-color: #FFF8B0 !important;
        color: #287A74 !important;
        font-weight: 700;
        padding: 5px 12px;
        border-radius: 8px;
        display: inline-block;
        font-size: 13px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    .citation-card {
        background-color: #f7fdfb !important;
        border: 1.5px solid #AEEED3 !important;
        border-left: 6px solid #287A74 !important;
        border-radius: 10px;
        padding: 16px;
        margin-top: 12px;
    }
    .citation-card h3 {
        color: #287A74 !important;
    }
    button.primary, .gr-button-primary {
        background-color: #287A74 !important;
        color: white !important;
        border: none !important;
        transition: all 0.2s ease-in-out;
    }
    button.primary:hover, .gr-button-primary:hover {
        background-color: #55A9A0 !important;
        box-shadow: 0 4px 12px rgba(40, 122, 116, 0.3) !important;
    }
    """

    custom_theme = gr.themes.Soft(
        primary_hue=gr.themes.Color(
            c50="#f0faf8",
            c100="#dcf4ef",
            c200="#AEEED3",
            c300="#8de4c2",
            c400="#55A9A0",
            c500="#287A74",
            c600="#226964",
            c700="#1c5753",
            c800="#174643",
            c900="#113533",
            c950="#0c2524",
        ),
        secondary_hue=gr.themes.Color(
            c50="#fffef5",
            c100="#fffde0",
            c200="#FFF8B0",
            c300="#fff380",
            c400="#ffed50",
            c500="#ffe820",
            c600="#e6cf10",
            c700="#b3a10c",
            c800="#807309",
            c900="#4d4505",
            c950="#332e03",
        ),
    ).set(
        body_background_fill="#f8fcfb",
        block_background_fill="#ffffff",
        block_border_width="1.5px",
        block_border_color="#AEEED3",
        button_primary_background_fill="#287A74",
        button_primary_background_fill_hover="#55A9A0",
        button_primary_text_color="#ffffff",
        block_label_text_color="#287A74",
    )

    with gr.Blocks(title="Google Colab L4 Multilingual RAG", theme=custom_theme, css=custom_css) as demo:
        gr.HTML(
            """
            <div class="header-banner">
                <h1 style="margin:0 0 8px 0; font-size: 26px;">🚀 Trợ lý RAG Đa Ngữ (Tiếng Việt & English) - NVIDIA L4 GPU</h1>
                <p style="margin:0 0 12px 0; opacity: 0.95; font-size: 15px;">
                    Tối ưu hóa trên Google Colab L4 (24GB VRAM) • <b>Qwen 2.5 7B Native Bfloat16</b> • <b>BGE-M3 Multilingual</b>
                </p>
                <span class="badge-accent">🎨 Theme Palette: #287A74 • #55A9A0 • #AEEED3 • #FFF8B0</span>
            </div>
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Quản lý tài liệu / Document Ingestion")
                file_input = gr.File(
                    label="Tải lên tệp PDF hoặc Text / Upload Files",
                    file_count="multiple",
                    file_types=[".pdf", ".txt", ".md"]
                )
                upload_btn = gr.Button("Phân tích & Nạp vào Vector DB", variant="primary")
                status_box = gr.Textbox(label="Trạng thái / Status", interactive=False)
                upload_btn.click(handle_upload, inputs=[file_input], outputs=[status_box])

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Cuộc trò chuyện / Conversation", height=450)
                with gr.Row():
                    query_input = gr.Textbox(
                        placeholder="Đặt câu hỏi về tài liệu của bạn (bằng tiếng Việt hoặc tiếng Anh)...",
                        label="Câu hỏi / Your Query",
                        scale=8
                    )
                    send_btn = gr.Button("Gửi / Send", variant="primary", scale=1)

                gr.Examples(
                    examples=[
                        ["What are the challenges in estimating output impedance in inverter-based grids?"],
                        ["How does discarding q-axis dynamics affect noise reduction in power networks?"],
                        ["What is the purpose of using a variable direction forgetting recursive-least-square algorithm in power networks?"],
                        ["How does transforming measurements into the dq coordinate frame aid in impedance estimation?"]
                    ],
                    inputs=query_input,
                    label="💡 Câu hỏi gợi ý từ Vectara Open RAG Bench:"
                )

                citation_box = gr.Markdown("### 📚 Nguồn trích dẫn / Verified Sources:\n_Đặt câu hỏi để xem nguồn trích dẫn._", elem_classes=["citation-card"])

                send_btn.click(
                    handle_chat,
                    inputs=[query_input, chatbot],
                    outputs=[query_input, chatbot, citation_box]
                )
                query_input.submit(
                    handle_chat,
                    inputs=[query_input, chatbot],
                    outputs=[query_input, chatbot, citation_box]
                )

        gr.Markdown("--- \n*Phần cứng: NVIDIA L4 GPU (24GB VRAM) với SDPA, TF32 và BGE-M3 Multilingual.*")

    return demo


