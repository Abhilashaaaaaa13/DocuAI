import streamlit as st
from utils import getvectorstore, llm, rerank_docs, rewrite_with_history
from graph_builder import build_knowledge_graph
import os
from self_expanding import try_expand_and_answer

st.set_page_config(page_title="BRAIN", layout="wide")

st.markdown("""
<style>
    .stChatInput {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        z-index: 999;
        background-color: #0e1117;
        padding: 12px 2rem 16px 2rem;
        border-top: 1px solid #2a2a2a;
    }
    .main .block-container {
        padding-bottom: 110px;
    }
</style>
""", unsafe_allow_html=True)

st.title("📚 RAG Brain")

tab1, tab2 = st.tabs(["📤 Upload & Map", "💬 Chat"])

# ─────────────────────────────────────────────────────
#  TAB 1: UPLOAD & KNOWLEDGE GRAPH
# ─────────────────────────────────────────────────────
with tab1:

    uploaded = st.file_uploader(
        "Upload a file",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        help="Supported: PDF (text or scanned), Word (.docx), Text (.txt), Images"
    )

    if uploaded:
        ext       = os.path.splitext(uploaded.name)[1].lower()
        file_path = f"temp_{uploaded.name}"

        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())

        if "vectorstore" not in st.session_state or st.session_state.get("current_file") != file_path:
            with st.spinner("🔍 Detecting file type and indexing…"):
                try:
                    retriever, file_type_label = getvectorstore(file_path)
                    st.session_state.vectorstore     = retriever
                    st.session_state.current_file    = file_path
                    st.session_state.file_path       = file_path
                    st.session_state.file_type_label = file_type_label
                    st.session_state.pop("graph_html_content", None)
                    st.session_state.pop("suggestions", None)
                    st.session_state.messages = []
                except ValueError as e:
                    st.error(f"❌ Could not extract content: {e}")
                    st.stop()

            st.success(f"✅ Indexed!  |  Type: **{st.session_state.file_type_label}**  |  File: **{uploaded.name}**")
        else:
            st.success(f"✅ Ready  |  Type: **{st.session_state.file_type_label}**  |  File: **{uploaded.name}**")

        # ── Knowledge Graph ──
        st.markdown("---")
        st.subheader("🧠 Knowledge Graph")

        col_btn, col_info = st.columns([1, 3])
        with col_btn:
            gen_graph = st.button("🚀 Generate Knowledge Graph", use_container_width=True)
        with col_info:
            st.caption("Extracts key topics & relationships from your document. Works for PDF, scanned PDF, DOCX, TXT and images.")

        if gen_graph:
            with st.spinner("🔬 Building graph… (may take ~30s for large files)"):
                try:
                    graph_path  = f"graph_{os.path.basename(file_path)}.html"
                    output_html = build_knowledge_graph(file_path, output_path=graph_path)
                    with open(output_html, "r", encoding="utf-8") as fh:
                        st.session_state.graph_html_content = fh.read()
                    st.success("✅ Graph built!")
                except Exception as e:
                    st.error(f"❌ Graph generation failed: {e}")

        if "graph_html_content" in st.session_state:
            st.markdown("**Tip:** Click a node to highlight connections · Search by name · Toggle physics to rearrange")
            st.components.v1.html(st.session_state.graph_html_content, height=650, scrolling=False)

        # ── Suggested Questions ──
        st.markdown("---")
        st.subheader("🔮 Suggested Questions")

        if st.button("✨ Generate Smart Questions"):
            with st.spinner("Generating questions…"):
                docs = st.session_state.vectorstore.invoke("main concepts discussed in this document")
                if docs:
                    prompt = f"""Generate EXACTLY 5 intelligent questions someone might ask after reading this document.

Document excerpt:
{docs[0].page_content[:1000]}

Return ONLY the 5 questions, one per line, no numbering or bullets.
"""
                    questions = llm.invoke(prompt).content.strip().split("\n")
                    st.session_state.suggestions = [
                        q.strip("0123456789.-) ").strip()
                        for q in questions if q.strip()
                    ][:5]

        if st.session_state.get("suggestions"):
            cols = st.columns(2)
            for idx, q in enumerate(st.session_state.suggestions):
                with cols[idx % 2]:
                    if st.button(f"💬 {q}", key=f"sug_{idx}"):
                        st.session_state.pending_query = q
                        st.rerun()


# ─────────────────────────────────────────────────────
#  TAB 2: CHAT
# ─────────────────────────────────────────────────────
with tab2:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "vectorstore" not in st.session_state:
        st.info("⬅️ Upload a file first in the Upload & Map tab.")
        st.stop()

    file_label = st.session_state.get("file_type_label", "Document")
    file_name  = os.path.basename(st.session_state.get("current_file", "")).replace("temp_", "")
    st.success(f"✅ {file_label}  |  **{file_name}**")

    # ── Chat history ──
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

    if st.session_state.get("suggestions"):
        st.subheader("💡 Try these questions")
        cols = st.columns(2)
        for idx, q in enumerate(st.session_state.suggestions):
            with cols[idx % 2]:
                if st.button(q, key=f"btn_{idx}"):
                    st.session_state.pending_query = q
                    st.rerun()

    user_input = st.chat_input("Ask anything about the document…")

    query = None
    if user_input:
        query = user_input
    elif st.session_state.get("pending_query"):
        query = st.session_state.pop("pending_query")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):

                rewritten = rewrite_with_history(query, st.session_state.messages[:-1])

                docs = st.session_state.vectorstore.invoke(rewritten)
                docs = docs[:8]
                docs = rerank_docs(rewritten, docs)
                docs = docs[:5]

                context = "\n\n".join(
                    f"--- Page {d.metadata.get('page', 'N/A')} ---\n{d.page_content[:800]}"
                    for d in docs
                )

                # ── Step 1: Get answer from document ──
                answer_prompt = f"""You are a document QA assistant.

Answer using the context provided. Be thorough and helpful.

RULES:
1. If the context contains relevant information, answer fully using it.
2. If the context has NO relevant info, reply ONLY with the exact phrase: NOT_IN_DOCUMENT
3. Do NOT fabricate facts not in the context.

Context:
{context}

Question:
{rewritten}

Answer:
"""
                answer = llm.invoke(answer_prompt).content.strip()

                # ── Step 2: Decide whether to go to web ──
                # Check both the sentinel phrase AND do an LLM coverage check
                sources  = []
                evidence = []
                expansion = {"expanded": False}

                needs_web = False

                if "NOT_IN_DOCUMENT" in answer.upper():
                    # Clear sentinel — document has nothing
                    needs_web = True
                    answer    = "NOT_IN_DOCUMENT"  # normalise for try_expand_and_answer

                elif not docs:
                    needs_web = True

                else:
                    # Secondary check: ask LLM if the answer is actually useful
                    coverage_prompt = f"""Does this answer actually address the question, or is it vague/incomplete?

Question: {rewritten}
Answer: {answer}

Reply with ONE word only:
GOOD  → answer directly addresses the question
POOR  → answer is vague, off-topic, or missing key info
"""
                    coverage = llm.invoke(coverage_prompt).content.strip().upper()
                    if "POOR" in coverage:
                        needs_web = True

                if needs_web:
                    # Pass a clear "not found" signal so self_expanding always searches
                    expansion = try_expand_and_answer(rewritten, "not in document")

                # ── Step 3: Display ──
                if expansion.get("expanded"):
                    final_answer    = expansion["answer"]
                    provider_badges = " + ".join(f"**{p.title()}**" for p in expansion["providers"])
                    st.markdown(f"**Answer (via {provider_badges}):**\n\n{final_answer}")
                    st.info("💡 Answer sourced from the web and saved to your knowledge base.")
                    sources = [u for u in expansion.get("sources", []) if u]
                    if sources:
                        st.markdown("**Sources:**")
                        for url in sources:
                            st.markdown(f"- {url}")
                else:
                    final_answer = answer
                    st.markdown(final_answer)

                for d in docs[:3]:
                    evidence.append({
                        "page":    d.metadata.get('page', 'N/A'),
                        "snippet": d.page_content[:200]
                    })
                if evidence:
                    with st.expander("📍 Evidence from document"):
                        for ev in evidence:
                            st.markdown(f"**Page {ev['page']}**")
                            st.markdown(f"> {ev['snippet']}...")
                            st.divider()

        st.session_state.messages.append({
            "role":     "assistant",
            "content":  final_answer,
            "sources":  sources,
            "evidence": evidence,
        })