"""
utils.py

Improvements:
  - getvectorstore() also infers + returns document domain
  - Better chunking for technical documents (smaller chunks, code-aware separators)
  - rewrite_with_history() uses full chat context to resolve ambiguous queries
"""
import os
import uuid
from dotenv import load_dotenv
from document_loader import smart_load
from langchain_groq import ChatGroq
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.cache import InMemoryCache
import langchain
from sentence_transformers import CrossEncoder
from self_expanding import infer_domain

load_dotenv()
langchain.cache = InMemoryCache()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"}
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda"
)

PERSIST_DIR    = "./chroma_db"
DOC_COLLECTION = "doc_store"


def getvectorstore(file_path: str) -> tuple:
    """
    Returns: (ensemble_retriever, file_type_label, domain)
    domain is a short tag like 'software_engineering', 'urban_farming'
    used to filter web store results to the right context.
    """
    docs = smart_load(file_path)
    if not docs:
        raise ValueError("No content could be extracted from the file.")

    # Infer domain from the first chunk of text
    sample_text = docs[0].page_content if docs else ""
    domain = infer_domain(sample_text)

    # Technical docs benefit from smaller chunks with less overlap confusion
    # We use tighter separators that respect code blocks and paragraphs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # smaller → more precise retrieval
        chunk_overlap=100,
        separators=["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    doc_id = str(uuid.uuid4())
    for chunk in chunks:
        page = chunk.metadata.get('page', 0)
        try:
            chunk.metadata['page'] = int(page) + 1
        except (ValueError, TypeError):
            chunk.metadata['page'] = 1
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["origin"] = "document"
        chunk.metadata["domain"] = domain   # tag chunks with domain too

    # Delete previous doc_store so old docs don't bleed in
    try:
        old = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings,
            collection_name=DOC_COLLECTION,
        )
        old.delete_collection()
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        chunks, embeddings,
        persist_directory=PERSIST_DIR,
        collection_name=DOC_COLLECTION,
    )
    vectorstore.persist()
    print(f"[VectorStore] {len(chunks)} chunks | domain='{domain}'")

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 60}
    )

    bm25   = BM25Retriever.from_documents(chunks)
    bm25.k = 8

    ensemble = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.4, 0.6]   # slightly favour semantic for technical content
    )
    return ensemble, "Document", domain


def rerank_docs(query: str, docs: list, top_k: int = 5) -> list:
    if not docs:
        return []
    pairs  = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def rewrite_with_history(query: str, messages: list, domain: str = "") -> str:
    """
    Resolve the query using conversation history AND document domain context.
    This prevents "containerization" being rewritten toward urban farming
    when the current document is about software engineering.
    """
    if not messages:
        return query

    history = "\n".join(
        f"{m['role']}: {m['content'][:300]}"
        for m in messages[-4:]   # last 4 turns for better context
    )

    domain_hint = f"\nThe current document is about: {domain}." if domain else ""

    words = query.strip().split()
    if len(words) >= 5:
        prompt = (
            f"You are helping a retrieval system resolve a question.{domain_hint}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Current question: {query}\n\n"
            f"If the question is already clear and self-contained, return it UNCHANGED.\n"
            f"If it has pronouns or refers to previous messages, resolve them.\n"
            f"Return ONLY the resolved question, nothing else:"
        )
    else:
        prompt = (
            f"You are helping a retrieval system understand a short follow-up question.{domain_hint}\n\n"
            f"Conversation history:\n{history}\n\n"
            f"Follow-up: {query}\n\n"
            f"Rewrite this as a complete standalone question in the context of: {domain or 'the document'}.\n"
            f"Return ONLY the question:"
        )

    return llm.invoke(prompt).content.strip()