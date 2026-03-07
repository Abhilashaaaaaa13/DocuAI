"""
app.py — Context-aware RAG Brain

The central intelligence lives in _route_query():
  - Builds a full context picture: document domain + chat history + retrieved chunks
  - Asks LLM ONE structured question: can the doc answer this, or do we need web?
  - Passes domain to web retrieval so old results from different topics never surface
  - Passes chat history to query rewriter so follow-ups are resolved correctly

Key session_state fields:
  vectorstore     → EnsembleRetriever for current doc
  doc_domain      → inferred domain tag (e.g. "software_engineering")
  messages        → full chat history [{role, content, sources, evidence}]
  current_file    → path to current uploaded file
"""
import streamlit as st
from utils import getvectorstore, llm, rerank_docs, rewrite_with_history
from graph_builder import build_knowledge_graph
from cache_manager import get_graph_cache
from self_expanding import try_expand_and_answer, get_web_vectorstore
import os

st.set_page_config(page_title="RAG Brain", layout="wide")

st.markdown("""
<style>
    .stChatInput {
        position: fixed; bottom: 0; left: 0; right: 0; z-index: 999;
        background-color: #0e1117;
        padding: 12px 2rem 16px 2rem;
        border-top: 1px solid #2a2a2a;
    }
    .main .block-container { padding-bottom: 110px; }
</style>
""", unsafe_allow_html=True)

st.title("📚 DOCUMIND")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌐 Web Knowledge Base")
    try:
        web_vs    = get_web_vectorstore()
        web_count = web_vs._collection.count()
        st.metric("Stored web chunks", web_count)
        st.caption(
            "Web search results are saved here, tagged by domain. "
            "Future similar questions reuse them without re-searching."
        )
    except Exception:
        st.caption("Web store: empty")

    st.markdown("---")
    st.markdown("### 🧠 Current Document")
    domain = st.session_state.get("doc_domain", "—")
    st.caption(f"**Domain:** `{domain}`")
    st.caption("All web results fetched while this doc is active are tagged with this domain.")

    st.markdown("---")
    gc = get_graph_cache().stats()
    st.metric("Graph cache entries", gc["entries"])
    if st.button("🗑️ Clear graph cache"):
        get_graph_cache().clear()
        st.success("Cleared.")

    st.markdown("---")
    st.markdown("""
**How routing works:**

1. Query resolved using chat history + domain
2. Document chunks retrieved & reranked
3. LLM classifies into 3 routes:
   - **DOCUMENT** → doc covers it → answer from doc
   - **WEB_RELATED** → related topic, not in doc → web search with domain context
   - **WEB_GENERAL** → unrelated topic → plain web search
4. Web results saved by domain → reused for future similar questions
""")


tab1, tab2 = st.tabs(["📤 Upload & Map", "💬 Chat"])


# ─────────────────────────────────────────────────────────────────────────────
#  CORE ROUTING LOGIC
# ─────────────────────────────────────────────────────────────────────────────

def _route_query(rewritten_query: str, doc_context: str, chat_history: str) -> str:
    """
    3-way router using full context awareness.

    Returns one of:
      "DOCUMENT"    → doc has a good answer, use it
      "WEB_RELATED" → topic is related to doc domain but not covered → web search
                      (e.g. doc is about Docker, user asks about Kubernetes)
      "WEB_GENERAL" → question is unrelated to the document entirely → web search

    Both WEB routes go to the web; the distinction is used to craft a better
    search query for WEB_RELATED (we add domain context to the search).
    """
    domain = st.session_state.get("doc_domain", "general")

    prompt = f"""You are a routing assistant for a document QA system.

Document domain: {domain}

Recent conversation:
{chat_history}

User question: {rewritten_query}

Retrieved document context (top chunks):
{doc_context[:2000]}

Classify the question into exactly ONE of these three categories:

DOCUMENT     — The retrieved context directly answers the question, OR
               the question asks to explain/summarise something the doc covers.

WEB_RELATED  — The question is about a concept RELATED to the document's domain
               but the specific answer is NOT in the retrieved context.
               Examples: doc covers Docker → user asks about Kubernetes;
                         doc covers urban farming → user asks about hydroponics costs;
                         doc covers Python → user asks about asyncio internals.

WEB_GENERAL  — The question has nothing to do with the document's domain.
               Examples: doc covers Docker → user asks about cricket scores.

Reply with ONE word only — DOCUMENT, WEB_RELATED, or WEB_GENERAL:"""

    verdict = llm.invoke(prompt).content.strip().upper()
    print(f"[Router] '{rewritten_query[:60]}' → {verdict}")

    if "DOCUMENT" in verdict:
        return "DOCUMENT"
    if "WEB_RELATED" in verdict:
        return "WEB_RELATED"
    return "WEB_GENERAL"


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1: UPLOAD & KNOWLEDGE GRAPH
# ─────────────────────────────────────────────────────────────────────────────
with tab1:

    uploaded = st.file_uploader(
        "Upload a document",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        help="PDF (text or scanned), Word, plain text, or images"
    )

    if uploaded:
        # ── Save to dedicated uploads/ folder, not the project root ──────
        uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")
        os.makedirs(uploads_dir, exist_ok=True)

        # Prefix with session ID so concurrent users never overwrite each other
        session_id = st.session_state.setdefault("session_id", os.urandom(4).hex())
        file_path  = os.path.join(uploads_dir, f"{session_id}_{uploaded.name}")

        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Auto-delete the PREVIOUS upload for this session (keep disk clean)
        prev_file = st.session_state.get("current_file")
        if prev_file and prev_file != file_path and os.path.exists(prev_file):
            try:
                os.remove(prev_file)
                print(f"[Cleanup] Removed old upload: {os.path.basename(prev_file)}")
            except Exception:
                pass

        # Only re-index if it's a new file
        if "vectorstore" not in st.session_state or st.session_state.get("current_file") != file_path:
            with st.spinner("🔍 Indexing document and inferring domain…"):
                try:
                    retriever, label, domain = getvectorstore(file_path)
                    st.session_state.vectorstore     = retriever
                    st.session_state.current_file    = file_path
                    st.session_state.file_type_label = label
                    st.session_state.doc_domain      = domain
                    st.session_state.messages        = []
                    st.session_state.pop("graph_html_content", None)
                    st.session_state.pop("suggestions", None)
                except Exception as e:
                    st.error(f"❌ Indexing failed: {e}")
                    st.stop()

            st.success(
                f"✅ Indexed | **{uploaded.name}** | "
                f"Domain: **{st.session_state.doc_domain}**"
            )
        else:
            st.success(
                f"✅ Ready | **{uploaded.name}** | "
                f"Domain: **{st.session_state.get('doc_domain','?')}**"
            )

        # ── Knowledge Graph ──────────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🧠 Knowledge Graph")
        c1, c2 = st.columns([1, 3])
        with c1:
            gen_graph = st.button("🚀 Generate Graph", use_container_width=True)
        with c2:
            st.caption("Domain-aware extraction — technical docs get technical entities. Cached for instant rebuild.")

        if gen_graph:
            with st.spinner("Building knowledge graph…"):
                try:
                    # Save graphs to graphs/ folder, session-prefixed, auto-clean old ones
                    graphs_dir  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "graphs")
                    os.makedirs(graphs_dir, exist_ok=True)

                    session_id  = st.session_state.get("session_id", "default")
                    graph_path  = os.path.join(graphs_dir, f"{session_id}_{os.path.basename(file_path)}.html")

                    # Delete previous graph for this session
                    prev_graph = st.session_state.get("current_graph_path")
                    if prev_graph and prev_graph != graph_path and os.path.exists(prev_graph):
                        try:
                            os.remove(prev_graph)
                            print(f"[Cleanup] Removed old graph: {os.path.basename(prev_graph)}")
                        except Exception:
                            pass

                    out = build_knowledge_graph(file_path, output_path=graph_path)
                    st.session_state.current_graph_path = graph_path

                    with open(out, "r", encoding="utf-8") as fh:
                        st.session_state.graph_html_content = fh.read()
                    st.success("✅ Graph ready!")
                except Exception as e:
                    st.error(f"❌ Graph failed: {e}")

        if "graph_html_content" in st.session_state:
            st.caption("Click a node to isolate it · Search box top-left · Toggle Physics to rearrange")
            st.components.v1.html(st.session_state.graph_html_content, height=650, scrolling=False)

        # ── Suggested Questions ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("🔮 Suggested Questions")
        if st.button("✨ Generate Questions"):
            with st.spinner("Generating…"):
                docs = st.session_state.vectorstore.invoke("main topics and concepts")
                if docs:
                    raw_qs = llm.invoke(
                        f"The document is about: {st.session_state.get('doc_domain','this topic')}.\n\n"
                        f"Generate exactly 5 specific, insightful questions a reader would ask.\n\n"
                        f"Document excerpt:\n{docs[0].page_content[:1200]}\n\n"
                        f"Return ONLY 5 questions, one per line, no numbering or bullets:"
                    ).content.strip().split("\n")
                    st.session_state.suggestions = [
                        q.strip("0123456789.-) ").strip()
                        for q in raw_qs if q.strip()
                    ][:5]

        if st.session_state.get("suggestions"):
            cols = st.columns(2)
            for idx, q in enumerate(st.session_state.suggestions):
                with cols[idx % 2]:
                    if st.button(f"💬 {q}", key=f"sug_{idx}"):
                        st.session_state.pending_query = q
                        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2: CHAT
# ─────────────────────────────────────────────────────────────────────────────
with tab2:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectorstore" not in st.session_state:
        st.info("⬅️ Upload a document in the first tab to begin.")
        st.stop()

    # Header shows current doc + domain
    file_name = os.path.basename(st.session_state.get("current_file", "")).replace("temp_", "")
    doc_domain = st.session_state.get("doc_domain", "unknown")
    st.success(f"✅ **{file_name}**  |  Domain: `{doc_domain}`")

    # ── Render history ────────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                st.markdown("**Sources:**")
                for url in msg["sources"]:
                    if url:
                        st.markdown(f"- {url}")
            if msg.get("evidence"):
                with st.expander("📍 Evidence from document"):
                    for ev in msg["evidence"]:
                        st.markdown(f"**Page {ev['page']}**")
                        st.markdown(f"> {ev['snippet']}...")
                        st.divider()

    # Suggested question buttons in chat
    if st.session_state.get("suggestions"):
        cols = st.columns(2)
        for idx, q in enumerate(st.session_state.suggestions):
            with cols[idx % 2]:
                if st.button(q, key=f"btn_{idx}"):
                    st.session_state.pending_query = q
                    st.rerun()

    # ── Input ─────────────────────────────────────────────────────────────────
    user_input = st.chat_input("Ask anything…")
    query      = user_input or st.session_state.pop("pending_query", None)

    if not query:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):

            domain   = st.session_state.get("doc_domain", "general")
            messages = st.session_state.messages

            # ── 1. Build chat history string for context ───────────────────
            chat_history = "\n".join(
                f"{m['role'].upper()}: {m['content'][:300]}"
                for m in messages[-6:]   # last 6 turns = 3 exchanges
                if m["role"] != "user" or m["content"] != query  # exclude current
            )

            # ── 2. Rewrite query using history + domain ────────────────────
            rewritten = rewrite_with_history(
                query,
                messages[:-1],   # history without current message
                domain=domain
            )
            if rewritten != query:
                print(f"[Rewrite] '{query}' → '{rewritten}'")

            # ── 3. Retrieve document chunks ────────────────────────────────
            docs = st.session_state.vectorstore.invoke(rewritten)
            docs = docs[:10]
            docs = rerank_docs(rewritten, docs, top_k=5)

            doc_context = "\n\n".join(
                f"--- Page {d.metadata.get('page', 'N/A')} ---\n{d.page_content[:800]}"
                for d in docs
            )

            # ── 4. Route: DOCUMENT / WEB_RELATED / WEB_GENERAL ────────────
            route    = _route_query(rewritten, doc_context, chat_history)
            sources  = []
            evidence = []

            if route == "DOCUMENT":
                # ── Answer from document ───────────────────────────────────
                answer = llm.invoke(
                    f"You are an expert assistant for a {domain} document.\n"
                    f"Answer using ONLY the retrieved context below.\n"
                    f"Be thorough and specific. Use technical terms correctly.\n"
                    f"If the context doesn't contain the specific answer, reply with "
                    f"exactly: NEEDS_WEB\n\n"
                    f"Context:\n{doc_context}\n\n"
                    f"Question: {rewritten}\n\nAnswer:"
                ).content.strip()

                # ── Escalate to web if doc answer is incomplete ────────────
                incomplete_signals = [
                    "NEEDS_WEB",
                    "no mention", "no information", "not mentioned",
                    "does not mention", "doesn't mention", "not found",
                    "not provided", "no further information",
                    "what's missing", "missing:", "cannot find",
                    "not covered", "not discussed", "not specified",
                ]
                needs_web = any(sig in answer.lower() for sig in incomplete_signals) \
                            or answer.strip().upper() == "NEEDS_WEB" \
                            or len(answer.strip()) < 40

                if needs_web:
                    print(f"[Escalate→Web] Doc answer incomplete for: '{rewritten}'")
                    route = "WEB_RELATED"   # treat as related web search
                else:
                    final_answer = answer
                    st.markdown(final_answer)

                    for d in docs[:3]:
                        evidence.append({
                            "page":    d.metadata.get("page", "N/A"),
                            "snippet": d.page_content[:220]
                        })
                    if evidence:
                        with st.expander("📍 Evidence from document"):
                            for ev in evidence:
                                st.markdown(f"**Page {ev['page']}**")
                                st.markdown(f"> {ev['snippet']}...")
                                st.divider()

            if route in ("WEB_RELATED", "WEB_GENERAL"):
                # ── Web fallback ───────────────────────────────────────────
                # WEB_RELATED: enrich query with domain so results are on-topic
                # WEB_GENERAL: plain search
                if route == "WEB_RELATED":
                    search_query = f"{rewritten} in {domain.replace('_', ' ')}"
                    badge_note   = f"related to your **{domain.replace('_', ' ')}** document"
                else:
                    search_query = rewritten
                    badge_note   = "general web search"

                print(f"[Web] route={route} | search='{search_query}'")
                expansion = try_expand_and_answer(search_query, domain)

                if expansion.get("expanded"):
                    final_answer = expansion["answer"]
                    from_cache   = expansion.get("from_cache", False)
                    providers    = " + ".join(
                        f"**{p.title()}**" for p in expansion.get("providers", ["Web"])
                    )
                    cache_badge = " *(web cache)*" if from_cache else " *(live search)*"

                    st.markdown(f"**Answer via {providers}{cache_badge}:**\n\n{final_answer}")
                    st.info(f"💡 Not in the document — sourced via {badge_note}. Saved for future queries.")

                    sources = [u for u in expansion.get("sources", []) if u]
                    if sources:
                        st.markdown("**Sources:**")
                        for url in sources:
                            st.markdown(f"- {url}")
                else:
                    final_answer = (
                        "I couldn't find an answer in the document or on the web. "
                        "Try rephrasing or check your internet connection."
                    )
                    st.markdown(final_answer)

    # ── Save to history ───────────────────────────────────────────────────────
    st.session_state.messages.append({
        "role":     "assistant",
        "content":  final_answer,
        "sources":  sources,
        "evidence": evidence,
    })