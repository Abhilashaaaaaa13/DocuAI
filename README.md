# 📚 DOCU AI

A context-aware document QA system that answers questions from your uploaded documents — and automatically searches the web for anything not covered, building a persistent, domain-tagged knowledge base as it goes.

---

## ✨ Features

### 1. Multi-Format Document Support
Upload any of the following and RAG Brain will extract and index it automatically:

| Format | How it's processed |
|---|---|
| PDF (text-based) | PyPDFLoader — fast direct extraction |
| PDF (scanned) | PyMuPDF + Tesseract OCR — page by page |
| Word (`.docx`) | Docx2txtLoader |
| Plain text (`.txt`) | TextLoader |
| Images (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`) | Pillow + Tesseract OCR |

---

### 2. Intelligent Document Indexing
When a file is uploaded, RAG Brain:
- Splits the document into optimised chunks (800 chars, 100 overlap)
- Embeds all chunks using `BAAI/bge-small-en-v1.5` via HuggingFace
- Stores embeddings in a **ChromaDB** collection (`doc_store`) isolated per upload
- Infers a **domain tag** (e.g. `software_engineering`, `urban_farming`, `finance`) from the document content using an LLM call
- Deletes and recreates the collection on new upload so old documents never pollute results

---

### 3. Hybrid Retrieval + Reranking
Every query goes through a two-stage retrieval pipeline:

**Stage 1 — Hybrid Retrieval (BM25 + Semantic MMR)**
- BM25 keyword retriever catches exact term matches
- ChromaDB MMR (Maximal Marginal Relevance) retriever catches semantic matches
- Results are combined with weights: 40% BM25, 60% semantic
- Fetches top 12 candidates

**Stage 2 — Cross-Encoder Reranking**
- All candidates are reranked using `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Top 5 chunks are kept for context

This two-stage approach is significantly more accurate than either BM25 or semantic retrieval alone.

---

### 4. Context-Aware Query Rewriting
Before retrieval, the query is resolved using the last 4 turns of chat history and the document's domain:

- Short follow-ups like *"tell me more"* or *"what about kafka?"* are expanded into fully standalone questions
- Pronouns like *"it"*, *"they"*, *"this approach"* are resolved to their referents
- The domain hint prevents ambiguous terms from being misinterpreted (e.g. *"containerization"* stays in the software context, not urban farming)

---

### 5. 3-Way Intelligent Router
The core decision engine. After retrieving document chunks, an LLM classifies every query into one of three routes:

```
DOCUMENT     → retrieved chunks directly answer the question
                → answer from document with page evidence

WEB_RELATED  → question is about a concept in the same domain
                but not covered in the document
                → web search with domain context appended to query
                  e.g. "who invented kafka" → "who invented kafka in software engineering"

WEB_GENERAL  → question is completely unrelated to the document
                → plain web search
```

Additionally, even when routed to DOCUMENT, if the generated answer contains signals of incompleteness (`"not mentioned"`, `"not found"`, `"not covered"`, etc.) or is too short, it automatically **escalates to WEB_RELATED** rather than returning a partial answer.

---

### 6. Dynamic Self-Expanding Web Knowledge Base
When the router sends a query to the web:

1. **Check web store first** — searches the existing `web_store` ChromaDB collection for domain-matched chunks from previous searches
2. **Validate cached answer** — if cached chunks exist but produce a poor/vague answer, they are discarded and a live search is performed
3. **Live search** (DuckDuckGo + optional Tavily) — fetches fresh results, saves them to `web_store` tagged with the current domain
4. **Future reuse** — the next time a similar question is asked in the same domain, the answer is served from the web store without hitting the search API again

Web results are stored in a completely **separate ChromaDB collection** (`web_store`) from the document collection (`doc_store`). They never pollute document retrieval.

---

### 7. Domain-Tagged Web Store
Every web chunk saved to the store carries a `domain` metadata tag. When retrieving:

- Only chunks whose domain matches the current document's domain are returned
- This means urban farming search results from a previous session will never surface when you're asking about Docker

This allows the web store to grow indefinitely across different document sessions without any cross-contamination.

---

### 8. Interactive Knowledge Graph
Generate a visual knowledge graph of any uploaded document:

- LLM extracts `(subject, relation, object)` triples from each chunk
- Domain-aware extraction prompt — a software engineering doc produces `Container`, `Orchestration`, `Microservice` nodes, not generic `Concept/Other`
- Concepts are normalised (stem-based deduplication: *"Urban Farming"* and *"Urban Farm"* merge to one node)
- Visualised with **vis.js** — interactive, physics-based layout

Graph features:
- Click any node to isolate its connections
- Search box to highlight specific concepts
- Toggle physics on/off to rearrange layout
- Hover for full relationship details
- Colour-coded by category (technology, process, concept, person, etc.)

Graph extraction results are **cached in SQLite** — rebuilding the graph for the same document is instant.

---

### 9. AI-Generated Suggested Questions
After uploading, click **✨ Generate Questions** to get 5 domain-specific questions the LLM thinks would be most insightful to ask about the document. Questions appear as clickable buttons in both the Upload tab and the Chat tab.

---

### 10. Clean File Management
- Uploaded files are saved to `uploads/` (not the project root)
- Generated graphs are saved to `graphs/` (not the project root)
- Each browser session gets a random ID prefix — concurrent users never overwrite each other's files
- Previous upload/graph for a session is auto-deleted when a new file is uploaded
- ChromaDB embeddings and the graph cache are **never affected** by file cleanup

---

## 🏗️ Architecture

```
User query
    │
    ▼
rewrite_with_history()          ← resolves follow-ups using chat history + domain
    │
    ▼
EnsembleRetriever               ← BM25 (40%) + ChromaDB MMR (60%)
    │
    ▼
CrossEncoder rerank             ← top 5 chunks selected
    │
    ▼
_route_query()                  ← DOCUMENT / WEB_RELATED / WEB_GENERAL
    │
    ├── DOCUMENT ──────────────► LLM answer from doc context
    │       │                        │
    │       │                   incomplete? ──► escalate to WEB_RELATED
    │       ▼
    │   📍 Evidence shown
    │
    └── WEB_RELATED / WEB_GENERAL
            │
            ▼
        check_web_store()       ← domain-filtered cache lookup
            │
            ├── good cached answer? ──► return it  *(web cache)*
            │
            └── no / bad answer?
                    │
                    ▼
                DuckDuckGo + Tavily  ← live search
                    │
                    ▼
                save_to_web_store()  ← tagged with domain
                    │
                    ▼
                LLM answer from web context  *(live search)*
```

---

## 📁 Project Structure

```
smartrag/
├── app.py                  # Streamlit UI, routing logic, chat
├── utils.py                # Indexing, hybrid retrieval, reranking, query rewriting
├── self_expanding.py       # Web search, domain-tagged web store
├── graph_builder.py        # Knowledge graph extraction and rendering
├── cache_manager.py        # SQLite cache for graph LLM extractions
├── document_loader.py      # Smart file loader (PDF/DOCX/TXT/image/OCR)
├── requirements.txt
├── .env                    # API keys (never commit)
├── .gitignore
│
├── uploads/                # User-uploaded files (auto-managed, gitignored)
├── graphs/                 # Generated graph HTML files (auto-managed, gitignored)
├── chroma_db/              # ChromaDB vector store (doc_store + web_store)
└── cache/                  # SQLite graph extraction cache
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki) (required for scanned PDFs and images)
- A [Groq API key](https://console.groq.com) (free tier available)
- *(Optional)* A [Tavily API key](https://tavily.com) for better web search results

### Step 1 — Clone the repo

```bash
git clone https://github.com/yourname/smartrag.git
cd smartrag
```

### Step 2 — Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

If you don't have a `requirements.txt` yet, install manually:

```bash
pip install streamlit
pip install langchain langchain-groq langchain-community langchain-text-splitters
pip install chromadb
pip install sentence-transformers
pip install rank-bm25
pip install pymupdf
pip install pytesseract pillow
pip install python-docx docx2txt
pip install duckduckgo-search
pip install tavily-python          # optional
pip install python-dotenv
```

### Step 4 — Install Tesseract

**Windows:**
Download and run the installer from https://github.com/UB-Mannheim/tesseract/wiki
Default install path: `C:\Program Files\Tesseract-OCR\tesseract.exe`

**macOS:**
```bash
brew install tesseract
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt install tesseract-ocr
```

### Step 5 — Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here   # optional
```

> Get your Groq key at https://console.groq.com — it's free and fast.
> Get your Tavily key at https://tavily.com — 1000 free searches/month.

### Step 6 — Run

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🔑 API Keys

| Key | Required | Where to get | Free tier |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | https://console.groq.com | Yes — generous free limits |
| `TAVILY_API_KEY` | ❌ Optional | https://tavily.com | 1000 searches/month |

Without Tavily, web search uses DuckDuckGo only (no API key needed, slightly less reliable).

---

## 🚀 Usage

### Uploading a Document
1. Go to the **Upload & Map** tab
2. Upload any PDF, Word doc, text file, or image
3. The system indexes the document and infers its domain automatically
4. The domain is shown in the success banner — this controls how web search results are tagged

### Chatting
1. Switch to the **Chat** tab
2. Type any question about the document
3. The system will:
   - Answer from the document if the answer is there
   - Search the web automatically if it's not, but the topic is related
   - Tell you clearly which source the answer came from

### Generating a Knowledge Graph
1. In the **Upload & Map** tab, click **🚀 Generate Graph**
2. Wait ~30 seconds on first run (cached instantly on subsequent runs)
3. Interact with the graph — click nodes, search, toggle physics

### Resetting
To start completely fresh (new document domain, clear all web results):
```bash
rm -rf chroma_db/ cache/
```
Uploaded files and graphs are managed automatically — no manual cleanup needed.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` |
| Vector Store | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Reranking | CrossEncoder `ms-marco-MiniLM-L-6-v2` |
| Graph Rendering | vis.js |
| Web Search | DuckDuckGo + Tavily |
| OCR | Tesseract + PyMuPDF |
| Graph Cache | SQLite |
| Orchestration | LangChain |

---

## 🔒 Privacy & Data

- All document processing happens **locally** on your machine
- Uploaded files are stored in `uploads/` and auto-deleted when you upload a new file
- Embeddings are stored locally in `chroma_db/`
- Web search queries are sent to DuckDuckGo/Tavily APIs — no document content is sent to them
- Your Groq API key is used only for LLM calls (query rewriting, answering, routing, graph extraction)

---

## 📝 Notes

- On first run, `BAAI/bge-small-en-v1.5` and the CrossEncoder model will be downloaded (~100MB total) from HuggingFace and cached locally
- GPU is used automatically if available (`device: cuda`). Falls back to CPU if not
- The graph cache lives in `cache/graph/graph_cache.db` — safe to delete if you want fresh extractions
- If you change documents frequently, run `rm -rf chroma_db/` periodically to keep the vector store lean