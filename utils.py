import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.cache import InMemoryCache
import langchain
from sentence_transformers import CrossEncoder
import torch
from functools import lru_cache

load_dotenv()
langchain.cache = InMemoryCache()

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

embeddings= HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs = {"device":"cuda"})

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device="cuda"
)

def getvectorstore(pdf_path):
    
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    
    # Better chunk settings: smaller chunks for better precision, better overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Smaller chunks = better semantic relevance
        chunk_overlap=200,  # More overlap = better continuity
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    #create unique document id
    doc_id = str(uuid.uuid4())
    
    # Ensure page metadata is preserved
    for chunk in chunks:
        page = chunk.metadata.get('page',0)
        chunk.metadata['page']=int(page)+1
        chunk.metadata["doc_id"]=doc_id

    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./chroma_db"
    )
    vectorstore.persist()

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":10,
            "fetch_k":50
        }
    )
    #create bm25 retrievers
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k=6
    #hybrid 
    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25,vector_retriever],
        weights=[0.5,0.5]
    )
    return ensemble_retriever

def rerank_docs(query,docs,top_k=4):
    pairs = [(query,d.page_content)for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(scores,docs),
        key=lambda x:x[0],
        reverse=True
    )
    return [doc for _,doc in ranked[:top_k]]

def rewrite_query(query):
    prompt=f"""Rewrite the following question into a clear search query for retrieving relevant information from a technical document
Question:
{query}

Rewritten query:
"""
    return llm.invoke(prompt).content.strip()


