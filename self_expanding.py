"""
self_expanding.py — Context-aware dynamic web knowledge base

KEY INSIGHT: Web chunks are tagged with the document domain they were
fetched for (e.g. "software_engineering", "urban_farming").
When retrieving, we ONLY return chunks whose domain matches the current
document's domain — so urban farming web results never surface when
the user has switched to a technical document.

Flow:
  1. Each document upload sets a domain tag (inferred by LLM from first chunk)
  2. Web search results are saved with that domain tag
  3. check_web_store() filters by current domain before returning results
  4. Web retrieval always respects current context
"""
import os
import re
import uuid
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults, TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda"}
)
llm      = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
ddg_tool = DuckDuckGoSearchResults(num_results=6)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)

PERSIST_DIR    = "./chroma_db"
WEB_COLLECTION = "web_store"


# ── Domain inference ──────────────────────────────────

def infer_domain(doc_sample: str) -> str:
    """
    Ask LLM to produce a short domain tag for the document.
    e.g. "software_engineering", "urban_farming", "machine_learning",
         "finance", "biology", "history"
    This tag is stored with every web chunk fetched for this document.
    """
    prompt = (
        f"What is the domain/field of this document? "
        f"Reply with 1-3 lowercase words joined by underscores (e.g. urban_farming, "
        f"software_engineering, machine_learning, finance). "
        f"No explanation.\n\n"
        f"Text:\n{doc_sample[:800]}\n\nDomain:"
    )
    try:
        domain = llm.invoke(prompt).content.strip().lower()
        # Sanitise: keep only word chars and underscores
        domain = re.sub(r'[^\w]', '_', domain)[:40]
        print(f"[Domain] Inferred: '{domain}'")
        return domain
    except Exception:
        return "general"


# ── Search tools ──────────────────────────────────────

def get_tavily():
    key = os.getenv("TAVILY_API_KEY", "").strip()
    return TavilySearchResults(max_results=5) if key else None


def search_duckduckgo(query: str) -> list[dict]:
    try:
        raw     = ddg_tool.run(query)
        entries = re.findall(
            r'\[snippet:\s*(.*?),\s*title:\s*(.*?),\s*link:\s*(.*?)\]',
            raw, re.DOTALL
        )
        results = [
            {"snippet": s.strip(), "title": t.strip(), "link": l.strip(), "source": "duckduckgo"}
            for s, t, l in entries
        ]
        if not results:
            results = [{"snippet": raw[:2000], "title": "DuckDuckGo", "link": "", "source": "duckduckgo"}]
        print(f"[DDG] {len(results)} results")
        return results
    except Exception as e:
        print(f"[DDG Error] {e}")
        return []


def search_tavily(query: str) -> list[dict]:
    tavily = get_tavily()
    if not tavily:
        return []
    try:
        raw = tavily.run(query)
        if isinstance(raw, list):
            return [{"snippet": r.get("content", "")[:800], "title": r.get("title", ""),
                     "link": r.get("url", ""), "source": "tavily"} for r in raw]
        elif isinstance(raw, str):
            return [{"snippet": raw[:2000], "title": "Tavily", "link": "", "source": "tavily"}]
    except Exception as e:
        print(f"[Tavily Error] {e}")
    return []


def merge_results(ddg: list[dict], tavily: list[dict]) -> list[dict]:
    seen, merged = set(), []
    for r in ddg + tavily:
        key = r.get("link", "").strip() or r["snippet"][:80]
        if key not in seen:
            seen.add(key)
            merged.append(r)
    return merged


# ── Web vector store ──────────────────────────────────

def get_web_vectorstore() -> Chroma:
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
        collection_name=WEB_COLLECTION,
    )


def save_to_web_store(results: list[dict], query: str, domain: str) -> None:
    """Save web results tagged with the current document's domain."""
    docs = []
    for r in results:
        docs.append(Document(
            page_content=f"{r['title']}\n\n{r['snippet']}",
            metadata={
                "source":   r.get("link") or r["source"],
                "provider": r["source"],
                "query":    query,
                "domain":   domain,          # ← domain tag for filtered retrieval
                "doc_id":   str(uuid.uuid4()),
                "page":     "web",
                "origin":   "web",
            }
        ))
    chunks = splitter.split_documents(docs)
    if not chunks:
        return
    # propagate domain to all chunks (splitter may lose it)
    for chunk in chunks:
        chunk.metadata["domain"] = domain
    vs = get_web_vectorstore()
    vs.add_documents(chunks)
    vs.persist()
    print(f"[Web Store] Saved {len(chunks)} chunks | domain='{domain}' | query='{query}'")


def check_web_store(query: str, domain: str) -> list[Document]:
    """
    Retrieve web chunks that match BOTH the query AND the current domain.
    This prevents urban_farming results from appearing when user is asking
    about software_engineering concepts.
    """
    try:
        vs      = get_web_vectorstore()
        # Fetch more candidates then filter by domain
        results = vs.similarity_search(query, k=10)
        domain_matches = [
            d for d in results
            if d.metadata.get("domain", "") == domain
               and d.metadata.get("origin") == "web"
        ]
        print(f"[Web Store] Query='{query}' domain='{domain}' → "
              f"{len(domain_matches)}/{len(results)} domain-matched chunks")
        return domain_matches[:5]
    except Exception as e:
        print(f"[Web Store Check Error] {e}")
        return []


# ── Main entry point ──────────────────────────────────

BAD_ANSWER_SIGNALS = [
    "could not find a reliable answer",
    "no information",
    "not mentioned",
    "not found",
    "no relevant",
    "cannot find",
    "doesn't mention",
    "does not mention",
    "not available",
    "not provided",
    "i don't know",
    "unclear",
]


def _is_good_answer(answer: str) -> bool:
    if not answer or len(answer.strip()) < 30:
        return False
    low = answer.strip().lower()
    return not any(sig in low for sig in BAD_ANSWER_SIGNALS)


def try_expand_and_answer(query: str, domain: str) -> dict:
    """
    1. Check web_store for domain-matched cached chunks
    2. If cached answer is GOOD → return it (fast path)
    3. If cached answer is bad OR no cache → live web search → save → answer
    """
    # ── Step 1: check existing domain-matched web chunks ──
    existing = check_web_store(query, domain)
    if existing:
        print(f"[Web Store] Found {len(existing)} cached chunks for domain='{domain}'")
        context = "\n\n".join(
            f"[{d.metadata.get('provider','web').upper()}] {d.metadata.get('source','')}\n"
            f"{d.page_content[:600]}"
            for d in existing
        )
        cached_answer = _answer_from_context(query, context)

        if _is_good_answer(cached_answer):
            print(f"[Web Store] Cache hit — good answer returned")
            sources = list({d.metadata.get("source", "") for d in existing if d.metadata.get("source")})
            return {
                "expanded":   True,
                "answer":     cached_answer,
                "sources":    sources,
                "providers":  list({d.metadata.get("provider", "web") for d in existing}),
                "from_cache": True,
            }
        else:
            print(f"[Web Store] Cache miss (poor answer) — falling through to live search")

    # ── Step 2: live web search ──
    print(f"[Web Search] Live search | domain='{domain}' | query='{query}'")
    ddg_results    = search_duckduckgo(query)
    tavily_results = search_tavily(query)
    all_results    = merge_results(ddg_results, tavily_results)

    if not all_results:
        return {"expanded": False, "answer": "", "sources": [], "providers": [], "from_cache": False}

    save_to_web_store(all_results, query, domain)

    context = "\n\n".join(
        f"[{r['source'].upper()}] {r.get('link', '')}\n{r['snippet'][:600]}"
        for r in all_results
    )
    answer    = _answer_from_context(query, context)
    sources   = [r["link"] for r in all_results if r.get("link")]
    providers = list({r["source"] for r in all_results})

    return {
        "expanded":   True,
        "answer":     answer,
        "sources":    sources,
        "providers":  providers,
        "from_cache": False,
    }


def _answer_from_context(query: str, context: str) -> str:
    prompt = (
        f"You are an AI research assistant. "
        f"Use ONLY the web context below to answer the question clearly and concisely.\n"
        f"Focus on answering exactly what was asked.\n"
        f"If the context genuinely doesn't contain the answer, say: Could not find a reliable answer.\n\n"
        f"Web Context:\n{context}\n\n"
        f"Question: {query}\n\nAnswer:"
    )
    return llm.invoke(prompt).content.strip()