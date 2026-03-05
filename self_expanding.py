"""Self-expading knowledge base for rag system
1. DuckDuckGo - no API key, always available
2. Tavily - free tier (1000 req/month), richer snippets 

Both results are merged, deduplicated, added to ChromaDB,
and used to generate a grounded answer.
 """
import os
import uuid
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import DuckDuckGoSearchResults, TavilySearchResults
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import re

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device":"cuda"}
)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

#search tools
ddg_tool = DuckDuckGoSearchResults(num_results=5)

def get_tavily():
    """Returns a TavilySearchResults instance only if the key exists"""
    key = os.getenv("TAVILY_API_KEY","").strip()
    if not key:
        return None
    return TavilySearchResults(max_results=5)

#splitter
splitter= RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n", "\n", ". ", " ", ""]
)
PERSIST_DIR="./chroma_db"
WEB_SOURCE_TAG ="web_search"

#confidence check
def is_answer_insufficient(answer:str)->bool:
    low_confidence_phrases = [
         "not in document", "not found", "no information",
        "cannot find", "doesn't mention", "does not mention",
        "not mentioned", "not available", "not provided",
        "i don't know", "i do not know", "unclear",
        "no relevant"
    ]
    return any(phrase in answer.lower() for phrase in low_confidence_phrases)

# duckduckgo search
def search_duckduckgo(query:str)->list[dict]:
    try:
        raw = ddg_tool.run(query)
        entries = re.findall( r'\[snippet:\s*(.*?),\s*title:\s*(.*?),\s*link:\s*(.*?)\]',
                               raw,
                               re.DOTALL)
        results=[{"snippet":s.strip(),"title":t.strip(),"link":l.strip(),"source":"duckduckgo"}
                 for s,t,l in entries]
        if not results:
            results = [{"snippet":raw[:2000],"title":"DuckDuckGo Result","link":"","source":"duckduckgo"}]
        print(f"[DDG] Got {len(results)} results")
        return results
    except Exception as e:
        print(f"[DDG Error] {e}")
        return []
    
#tavily search
def search_tavily(query:str)->list[dict]:
    tavily = get_tavily()
    if not tavily:
        return []
    try:
        raw_results = tavily.run(query)
        results = []
        if isinstance (raw_results,list):
            for r in raw_results:
                results.append({
                    "snippet":r.get("content","")[:800],
                    "title":r.get("title","Tavily Result"),
                    "link": r.get("url",""),
                    "source":"tavily",
                })
            return results
        elif isinstance(raw_results,str):
            results = [{"snippet":raw_results[:2000],"title":"Tavily Result","link":"","source":"tavily"}]
            print(f" [Tavily] got {len(results)} results")
            return results
        
    except Exception as e:
        print(f"[Tavily Error] {e}")
        return []
    
#merge+duplicate
def merge_results(ddg: list[dict], tavily: list[dict])-> list[dict]:
    seen_links = set()
    merged = []
    for r in ddg+tavily:
        link =r.get("link","").strip()
        key = link if link else r["snippet"][:80]
        if key not in seen_links:
            seen_links.add(key)
            merged.append(r)
    return merged

#results->langchain documents
def results_to_docs(results:list[dict],query:str)->list[Document]:
    docs=[]
    for r in results:
        content = f"{r['title']}\n\n{r['snippet']}"
        docs.append(Document(
            page_content=content,
            metadata={
                "source": r["link"] or r["source"],
                "title":r["title"],
                "origin": WEB_SOURCE_TAG,
                "provider":r["source"],
                "query":query,
                "doc_id": str(uuid.uuid4()),
                "page":"web",
            },
        ))
    return docs

#persist to chroma db
def add_to_vectorstore(docs:list[Document])->None:
    chunks = splitter.split_documents(docs)
    if not chunks:
        return
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    vectorstore.add_documents(chunks)
    vectorstore.persist()
    print(f"[KB Expand] Saved {len(chunks)} new chunks to ChromaDB ")

#generate ans from web content
def answer_from_web(query:str, web_docs: list[Document])->str:
    context = "\n\n".join(
        f"--[{d.metadata.get('provider','web').upper()}] {d.metadata.get('source','?')} --\n"
        f"{d.page_content[:600]}"
        for d in web_docs
    )
    prompt= f"""You are an AI research assistant.
    The user's document did not contain ans answer , so web search results are provided below.
    Use ONLY the web context to answer the question clearly and concisely. 
    If you still cant answer,say:"Could not find a reliable answer."
    Web Context:
    {context}
    Question:
    {query}
    Answer:"""
    return llm.invoke(prompt).content.strip()

#public entry point
def try_expand_and_answer(query:str, original_answer:str)->dict:
    """Call this after every RAG answer.
    Returns:
    {"expanded:bool,    #true ->web was searched
     "answer": str,            #updated answer
     "sources":list[str],      #URLS used
     "providers":list[str],    #which tools fired("duckduckgo",/"tavily")

    }"""
    if not is_answer_insufficient(original_answer):
        return {"expanded": False, "answer": original_answer,"sources":[], "providers":[]}
    print(f"[KB Expand] Triggering dual web search for: {query}")

    ddg_results = search_duckduckgo(query)
    tavily_results = search_tavily(query)
    all_results = merge_results(ddg_results,tavily_results)

    if not all_results:
        return {"expanded":False, "answer":original_answer,"sources":[], "providers":[]}
    web_docs = results_to_docs(all_results,query)
    new_answer = answer_from_web(query,web_docs)
    add_to_vectorstore(web_docs)

   
    sources= [r["link"] for r in all_results if r.get("link")]
    providers = list({r["source"] for r in all_results})

    return {
        "expanded":True,
        "answer": new_answer,
        "sources":sources,
        "providers":providers,
    }



        



