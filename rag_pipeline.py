import os
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json


print("Loading RAG assets...")
try:
    index = faiss.read_index('rag_assets/quotes.index')
    df = pd.read_csv('rag_assets/quotes_data.csv')
    df['tags'] = df['tags'].apply(lambda x: eval(x) if isinstance(x, str) else [])
   
    model = SentenceTransformer('all-MiniLM-L6-v2')   # Load the all-MiniLM-L6-v2 model
    print("Assets loaded successfully.")
except FileNotFoundError:
    print("Error: RAG assets not found. Please run 'prepare_data.py' first.")
    exit()

def retrieve_quotes(query: str, k: int = 5):
    """
    Performs a single-stage search to retrieve relevant documents.
    """
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)
    
    valid_indices = [i for i in indices[0] if i < len(df)]
    if not valid_indices:
        return []
        
    retrieved_docs = df.iloc[valid_indices].to_dict(orient='records')
    for i, doc in enumerate(retrieved_docs):
        doc['similarity_score'] = 1 / (1 + distances[0][i])
        
    return retrieved_docs

def generate_mock_answer(query: str, context_docs: list):
    """
    Simulates a high-quality response from an LLM API without making a real call.
    """
    print("--- Using MOCK API Response ---")
    if not context_docs:
        return {"summary": "I couldn't find any relevant quotes for your query.", "relevant_quotes": []}
        
    best_doc = context_docs[0]
    author = best_doc.get("author", "an author")
    summary = f"Based on the retrieved context for the query '{query[:30]}...', the most relevant quote appears to be from {author}."

    relevant_quotes = [
        {"quote": doc.get("quote"), "author": doc.get("author"), "tags": doc.get("tags")}
        for doc in context_docs
    ]
    
    return {"summary": summary, "relevant_quotes": relevant_quotes}

def query_rag_pipeline(query: str, author_filter: str = None):
    """
    The main pipeline function that orchestrates retrieval and mock generation.
    """
    retrieved_docs = retrieve_quotes(query)
    
  
    if author_filter:
        retrieved_docs = [
            doc for doc in retrieved_docs 
            if author_filter.lower() in doc.get('author', '').lower()
        ]

    generated_json = generate_mock_answer(query, retrieved_docs)
    
    return {"structured_answer": generated_json, "source_documents": retrieved_docs}