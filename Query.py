"""PrimeRAG query engine with a binary input guardrail.

The guardrail makes a single decision per turn:
    - DOCUMENT: the query is about the user's indexed PDFs  -> hybrid RAG
    - GENERAL : anything else (world knowledge, coding, chat) -> local LLM

Personal-memory and conversation persistence are intentionally not implemented.
If a query asks about personal history, the general prompt is instructed to say
so honestly instead of fabricating an answer.
"""

import os
import re
from typing import Any

import chromadb
import torch

os.environ["NLTK_DATA"] = os.path.expanduser("~/nltk_data")

from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.prompts import PromptTemplate
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

print("[Query] Initializing PrimeRAG query engine...")

# ---------------------------------------------------------------------------
# 1. Configuration and local models
# ---------------------------------------------------------------------------
DB_PATH = "./storage/vectorDB"
DOCSTORE_PATH = "./storage/docstore.json"
COLLECTION_NAME = "CORE"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Must match the embedding model used by Ingest.py.
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    device=device,
)

Settings.llm = Ollama(
    model="qwen2.5-coder:1.5b",
    base_url="http://localhost:11434",
    request_timeout=180.0,
    context_window=3072,
    temperature=0.1,
)
llm: Ollama = Settings.llm

# ---------------------------------------------------------------------------
# 2. Load the existing PDF index
# ---------------------------------------------------------------------------
chroma_client = chromadb.PersistentClient(path=DB_PATH)

try:
    chroma_collection = chroma_client.get_collection(name=COLLECTION_NAME)
except Exception as exc:
    raise RuntimeError(
        f"[Query] Chroma collection '{COLLECTION_NAME}' not found at '{DB_PATH}'. "
        "Run Ingest.py first."
    ) from exc

vectorstore = ChromaVectorStore(chroma_collection=chroma_collection)

if not os.path.exists(DOCSTORE_PATH):
    raise FileNotFoundError(
        f"[Query] Docstore not found at '{DOCSTORE_PATH}'. Run Ingest.py first."
    )

docstore = SimpleDocumentStore.from_persist_path(DOCSTORE_PATH)
if not docstore.docs or chroma_collection.count() == 0:
    raise ValueError(
        "[Query] The PDF index is empty. Add PDFs to storage/data/ and run Ingest.py."
    )

index = VectorStoreIndex.from_vector_store(vector_store=vectorstore)
top_k = min(10, chroma_collection.count())

dense_retriever = index.as_retriever(similarity_top_k=top_k)
keyword_retriever = BM25Retriever.from_defaults(
    docstore=docstore,
    similarity_top_k=top_k,
)

hybrid_retriever = QueryFusionRetriever(
    retrievers=[dense_retriever, keyword_retriever],
    similarity_top_k=top_k,
    num_queries=1,
    mode="reciprocal_rerank",
    use_async=False,
)

reranker = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-6-v2",
    top_n=min(3, top_k),
    device=device,
)

rag_prompt = PromptTemplate(
    "You are a strict research assistant. Answer using ONLY the provided context. "
    "Do not invent facts. Cite supported claims using [1], [2], etc., matching "
    "the order of the supplied context. If the answer is not in the context, "
    "state: 'I do not have enough information in the indexed documents.'\n\n"
    "Context:\n---------------------\n{context_str}\n---------------------\n"
    "Question: {query_str}\nAnswer: "
)

rag_query_engine = RetrieverQueryEngine.from_args(
    retriever=hybrid_retriever,
    node_postprocessors=[reranker],
    text_qa_template=rag_prompt,
)

print(f"[Query] PDF retrieval enabled: {chroma_collection.count()} indexed chunks.")

# ---------------------------------------------------------------------------
# 3. Input guardrail: DOCUMENT vs GENERAL
# ---------------------------------------------------------------------------
# Phrase-based cues rather than single words ("file", "page", "table"), so a
# general coding or science question is not mistaken for a document query.
DOCUMENT_CUES = (
    "uploaded pdf", "uploaded document", "uploaded file", "uploaded report",
    "this pdf", "that pdf", "my pdf", "in the pdf", "from the pdf",
    "this document", "that document", "my document",
    "in this document", "in the document", "according to the document",
    "the uploaded", "my files", "from my files", "my reports",
    "according to the report", "in the report", "from the report",
    "from the file", "in the file",
    "summarize the pdf", "summarise the pdf",
    "summarize this document", "summarise this document",
    "summarize this pdf", "summarise this pdf",
    "what does the pdf say", "what does this document say",
    "according to my pdf", "based on the uploaded",
    "indexed document", "indexed pdf", "my documents", "my pdfs",
)


def normalise(text: str) -> str:
    """Lowercase and collapse whitespace so cue matching is stable."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def classify_input(query: str) -> tuple[bool, str]:
    """Guardrail: return (is_document_query, reason).

    Rules handle explicit references first. Ambiguous queries fall through to a
    small local LLM call with a strict two-label prompt. Anything still
    unresolved defaults to GENERAL so a private-document path is never taken by
    accident.
    """
    q = normalise(query)

    if any(cue in q for cue in DOCUMENT_CUES):
        return True, "Matched an explicit document cue."

    prompt = f"""You are an input guardrail for a local PDF assistant.
Decide whether the user's query is about their own indexed PDFs and uploaded
documents, or whether it is a general question.

DOCUMENT: the user wants information from their indexed PDFs or uploaded files.
GENERAL: general knowledge, coding, math, science, greetings, or any query that
does not require the user's PDFs.

Do not choose DOCUMENT just because a general technical question mentions a
table, page, file, report, or document as a generic concept.
Return only one word: DOCUMENT or GENERAL.

Query: {query}

Label:"""

    try:
        raw = str(llm.complete(prompt)).strip().upper()
        found_doc = "DOCUMENT" in raw
        found_gen = "GENERAL" in raw
        if found_doc and not found_gen:
            return True, "LLM classified as a document query."
        if found_gen and not found_doc:
            return False, "LLM classified as a general query."
        print(f"[Guardrail] Ambiguous classifier output: {raw!r}")
    except Exception as exc:
        print(f"[Guardrail] Classifier failed: {exc}")

    return False, "Defaulted to a general query."


# ---------------------------------------------------------------------------
# 4. Query execution
# ---------------------------------------------------------------------------
def source_description(node_with_score: Any) -> str:
    metadata = node_with_score.node.metadata or {}
    filename = metadata.get("file_name") or metadata.get("filename")
    source = metadata.get("source") or metadata.get("file_path")
    if not filename and source:
        filename = os.path.basename(str(source))
    filename = str(filename or "Unknown source")
    page = metadata.get("page_label") or metadata.get("page_number") or metadata.get("page")
    return f"{filename} (Page {page})" if page is not None else filename


def general_answer(query: str) -> str:
    prompt = f"""You are PrimeRAG, a local assistant. Answer the user's general
question clearly and accurately. Do not claim to know personal facts, uploaded
document contents, or previous conversations. If the question requires that
kind of information, say so honestly instead of guessing.

User question: {query}

Answer:"""
    try:
        return str(llm.complete(prompt)).strip()
    except Exception as exc:
        return f"The local Ollama model could not answer: {exc}"


def execute_query(query: str) -> tuple[str, str, str, list[Any]]:
    """Return (label, reason, answer, source_nodes)."""
    is_document, reason = classify_input(query)

    if not is_document:
        return "GENERAL", reason, general_answer(query), []

    response = rag_query_engine.query(query)
    answer = str(response)
    source_nodes = list(response.source_nodes or [])
    if not source_nodes:
        answer = "I do not have enough information in the indexed documents to answer that."
    return "DOCUMENT", reason, answer, source_nodes


# ---------------------------------------------------------------------------
# 5. Interactive console
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- PrimeRAG Interactive Console ---")
    print("Type 'exit' or 'quit' to stop.")
    print("Guardrail: DOCUMENT (your PDFs) or GENERAL (everything else)")

    while True:
        try:
            user_query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Query] Interrupted.")
            break

        if user_query.lower() in {"exit", "quit"}:
            break
        if not user_query:
            continue

        print("\n[PrimeRAG] Processing...")
        label, reason, answer, source_nodes = execute_query(user_query)
        print(f"[Guardrail] -> {label} | {reason}")
        print(f"\n--- ANSWER ---\n{answer}")

        if source_nodes:
            print("\n--- REFERENCES ---")
            for index, node in enumerate(source_nodes, start=1):
                print(f"[{index}] {source_description(node)}")

    print("[Query] Goodbye.")