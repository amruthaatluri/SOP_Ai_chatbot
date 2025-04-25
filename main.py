import os
import argparse
import streamlit as st
from src.process_docx import process_all_documents
from src.chunk_text import process_extracted
from src.vector_store import process_and_store_vectors
from src.chatbot import chatbot

def run_processing():
    """Runs document extraction, chunking, and vectorization."""
    print("\n Extracting text from DOCX files...\n")
    process_all_documents("data/raw", "data/processed")
    print("\n Text extraction complete! Extracted files are in 'data/processed'.")

    print("\n Chunking extracted text...\n")
    process_extracted("data/processed", "data/chunked", max_tokens=512)
    print("\n Chunking complete! Chunked files are in 'data/chunked'.")

    print("\n Generating embeddings with FAISS...\n")
    process_and_store_vectors()
    print("\n All steps complete! Data is ready for chatbot retrieval.")

def run_chatbot():
    """Runs the chatbot UI using Streamlit."""
    st.title("🤖 AI Chatbot: Llama 3 + FAISS-Powered Retrieval")

    query = st.text_input("Ask a question:", "")

    if st.button("Get Answer"):
        if query.strip():
            with st.spinner("Searching and generating response..."):
                response = chatbot(query)
                st.success(response)
        else:
            st.warning("Please enter a question.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI Chatbot or Process Documents")
    parser.add_argument("--process", action="store_true", help="Run document processing (extraction, chunking, vectorization)")
    parser.add_argument("--chatbot", action="store_true", help="Run chatbot UI")

    args = parser.parse_args()

    if args.process:
        run_processing()
    elif args.chatbot:
        run_chatbot()
    else:
        print("\n No valid argument provided. Use '--process' to process documents or '--chatbot' to start the chatbot.")
