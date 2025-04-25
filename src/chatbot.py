
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from collections import deque

#  Load embedding model (Ensure it's the same as used in vector_store.py)
model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Load FAISS index and metadata
index = faiss.read_index("data/vectors/faiss_index.bin")
with open("data/vectors/metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Context history for multi-turn conversations
chat_history = deque(maxlen=3)  # Stores the last 3 exchanges for context chaining

def generate_expanded_queries(query):
    """Uses Llama 3 to generate five expanded queries related to the input query."""
    prompt = f"""
    You are a search optimization assistant. Generate 5 different reworded versions of the following query while keeping the same intent:
    
    Query: "{query}"

    Ensure the variations cover different phrasing but do not change the original meaning.
    Output them as a numbered list.
    """
    
    response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])

    expanded_queries = response["message"]["content"].strip().split("\n")
    expanded_queries = [q.replace(f"{i+1}.", "").strip() for i, q in enumerate(expanded_queries) if q.strip()]

    print(f"Expanded Queries: {expanded_queries}")  #  Debugging output
    return expanded_queries[:5]  # Limit to 5 queries

def search_faiss(query, top_k=3):
    """Finds the most relevant text chunks using LLM-generated query expansion and FAISS similarity search."""
    expanded_queries = generate_expanded_queries(query)

    all_results = []
    for exp_query in expanded_queries:
        query_embedding = model.encode([exp_query], convert_to_numpy=True)
        _, indices = index.search(query_embedding, top_k)

        for idx in indices[0]:
            if idx < len(metadata):
                doc = metadata[idx]
                all_results.append({
                    "title": doc["title"],
                    "content": doc.get("text", "No content found.")  #  Fetch actual text content
                })

    # Remove duplicates while preserving order
    unique_results = {doc["content"]: doc for doc in all_results}.values()
    return list(unique_results)

def generate_response_with_ollama(query, retrieved_docs, model_name="llama3"):

    if not retrieved_docs:
        return "I couldn't find any relevant information on this topic."

    #  Maintain chat history for context chaining
    context = "\n\n".join([f"Title: {doc['title']}\nContent: {doc['content']}" for doc in retrieved_docs])
    chat_history.append(f"User: {query}\nBot: {context}")

    prompt = f"""
    You are an AI assistant with access to the following document information.
    
    Context:
    {context}

    Chat History:
    {''.join(chat_history)}

    Answer the user's question using only the provided context. If the answer isn't in the context, say "I don't know."

    User's Question: {query}
    Answer:
    """

    response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])

    return response["message"]["content"]

def chatbot(query):
    """Handles user queries, retrieves relevant information, and generates a response."""
    retrieved_docs = search_faiss(query)
    response = generate_response_with_ollama(query, retrieved_docs, model_name="llama3")  
    return response
