import streamlit as st
from utils import getvectorstore, llm, rewrite_query,rerank_docs
from graph_builder import build_knowledge_graph
import os
from self_expanding import try_expand_and_answer

st.set_page_config(page_title="BRAIN", layout="wide")
st.title("📚 RAG Brain")

tab1, tab2 = st.tabs(["📤 Upload & Map", "💬 Chat"])

#  TAB 1: UPLOAD & MAP
with tab1:

    uploaded = st.file_uploader("Upload PDF", type="pdf")

    if uploaded:
        pdf_path = f"temp_{uploaded.name}"

        with open(pdf_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Index document
        if "vectorstore" not in st.session_state or st.session_state.get("current_pdf") != pdf_path:
            with st.spinner("Indexing document..."):
                st.session_state.vectorstore = getvectorstore(pdf_path)
                st.session_state.current_pdf = pdf_path
                st.session_state.pdf_path = pdf_path

            st.success("✅ Document indexed successfully!")

        # Knowledge graph
        if st.button("🚀 Generate Knowledge Map + Suggestions"):

            with st.spinner("Building Knowledge Graph..."):

                map_html = build_knowledge_graph(st.session_state.pdf_path)

                st.components.v1.html(
                    open(map_html, "r", encoding="utf-8").read(),
                    height=700
                )

                # Suggested Questions
                st.subheader("🔮 Suggested Smart Questions")

                docs = st.session_state.vectorstore.invoke(
                    "main concepts discussed in this document"
                )

                if docs:
                    prompt = f"""
Generate EXACTLY 5 intelligent questions someone might ask after reading this document.

Document excerpt:
{docs[0].page_content[:1000]}
"""

                    questions = llm.invoke(prompt).content.strip().split("\n")

                    st.session_state.suggestions = [
                        q.strip("- ").strip()
                        for q in questions
                        if q.strip()
                    ][:5]

                    for q in st.session_state.suggestions:
                        if st.button(q, key=f"sug_{hash(q)}"):
                            st.session_state.pending_query = q
                            st.rerun()


#  TAB 2: CHAT
with tab2:
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vectorstore" not in st.session_state:
        st.info("⬅️ Upload a PDF first")
        st.stop()

    st.success(f"✅ Document: **{st.session_state.current_pdf.replace('temp_', '')}**")
    
     #previous msg display
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    query = st.chat_input("Ask something about the document...")
    if query:
        st.session_state.messages.append({
            "role":"user",
            "content":query
        })
        with st.chat_message("user"):
            st.write(query)

    

    if query or st.session_state.get("pending_query"):

        if st.session_state.get("pending_query"):
            query = st.session_state.pending_query
            st.session_state.pending_query = None

        with st.spinner("Thinking with Groq..."):

            # Better retrieval: no rewriting
            rewritten = rewrite_query(query)
            
            docs = st.session_state.vectorstore.invoke(rewritten)
            #retreive more chunks
            docs = docs[:8]
            #rerank
            docs = rerank_docs(rewritten,docs)
            #keep best chunks
            docs =docs[:5]

            

            context = "\n\n".join(
                [
                    f"--- Page {d.metadata.get('page','N/A')} ---\n{d.page_content[:800]}"
                    for d in docs
                ]
            )

            answer_prompt = f"""
You are a document QA assistant.

Answer ONLY using the context provided.

STRICT RULES:
1. If the context does NOT contain the exact answer, reply exactly:
   Not in document
2. Do NOT infer or guess.
3. Do NOT use outside knowledge.

Context:
{context}

Question:
{rewritten}

Answer:
"""

            answer = llm.invoke(answer_prompt).content.strip()
            missing_info = ("not in document" in answer.lower() or "not mentioned" in answer.lower() or "not provided" in answer.lower() or "does not contain" in answer.lower())
            if missing_info:
                expansion = try_expand_and_answer(rewritten,answer)
            else:
                expansion = {"expanded":False}
            if expansion["expanded"]:
                answer = expansion["answer"]
                provider_badges = "+".join(f"**{p.title()}**" for p in expansion["providers"])
                st.session_state.messages.append({
                    "role":"assistant",
                    "content":answer
                })
                with st.chat_message("assistant"):
                    st.markdown(f"**Answer (via {provider_badges}):** {answer}")
                    st.info("Answer sourced from the web and saved to your knowledge base")

                    if expansion["sources"]:
                        st.markdown("**Sources:**")
                        for url in expansion["sources"]:
                            if url:
                                st.markdown(f"-{url}")
            else:
                st.session_state.messages.append({
                    "role":"assistant",
                    "content":answer
                })
                with st.chat_message("assistant"):
                    st.markdown(f"**Answer:** {answer}")
                    


            with st.expander("📍 Evidence from document"):

                for d in docs[:3]:
                    st.markdown(f"**Page {d.metadata.get('page','N/A')}**")
                    st.markdown(f"> {d.page_content[:200]}...")
                    st.divider()


        # Suggested question buttons
        if st.session_state.get("suggestions"):
            st.subheader("💡 Try these questions")

            cols = st.columns(2)

            for idx, q in enumerate(st.session_state.suggestions):
                with cols[idx % 2]:
                    if st.button(q, key=f"btn_{idx}"):
                        st.session_state.pending_query = q
                        st.rerun()