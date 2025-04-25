import os
import json
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

# Load Hugging Face model and tokenizer (replace 'sentence-transformers/all-MiniLM-L6-v2' with your choice)
model_name = "sentence-transformers/all-MiniLM-L6-v2"  # You can change to any Hugging Face model you prefer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_embeddings(text):
    """Generate embeddings using Hugging Face model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu().numpy()
    return embeddings

def load_chunks(input_folder="data/chunked"):
    """Loads chunked JSON files and returns text data with metadata."""
    text_chunks = []
    metadata = []

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(input_folder, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for chunk in data.get("chunks", []):
                text_chunks.append(chunk["text"])
                metadata.append({"title": chunk.get("title", "Untitled"), "filename": filename})

    return text_chunks, metadata

def build_faiss_index(text_chunks, metadata, output_folder="data/vectors"):
    """Generates embeddings using Hugging Face model and stores them in FAISS."""
    os.makedirs(output_folder, exist_ok=True)

    # Ensure text_chunks are not empty before encoding
    if not text_chunks:
        print("⚠️ No text data found for embeddings.")
        return

    # Generate embeddings for all chunks
    embeddings = np.vstack([get_embeddings(chunk) for chunk in text_chunks])

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save FAISS index
    faiss.write_index(index, os.path.join(output_folder, "faiss_index.bin"))

    # Save metadata including actual text content
    indexed_data = [{"title": metadata[i]["title"], "filename": metadata[i]["filename"], "text": text_chunks[i]} for i in range(len(text_chunks))]

    with open(os.path.join(output_folder, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(indexed_data, f, indent=4, ensure_ascii=False)

    print("\n FAISS index and metadata saved successfully!")

def process_and_store_vectors():
    """Loads chunked data, vectorizes it using Hugging Face model, and stores it in FAISS."""
    print("\n Generating embeddings using Hugging Face model and storing in FAISS...\n")

    text_chunks, metadata = load_chunks()
    
    if not text_chunks:
        print(" No text chunks found. Ensure 'data/chunked' has files.")
        return

    build_faiss_index(text_chunks, metadata)
    print("\n Vectorization complete! Embeddings stored successfully.")
