

import os
import json
import tiktoken  # OpenAI’s tokenizer for correct token counting
import re

def get_tokenizer():
    """Returns OpenAI's GPT-based tokenizer for accurate token counting."""
    return tiktoken.get_encoding("cl100k_base")

def count_tokens(text, tokenizer):
    """Counts tokens in a given text using OpenAI’s tokenizer."""
    return len(tokenizer.encode(text))

def split_into_paragraphs(text):
    """Splits text into paragraphs while preserving sentence structure."""
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

def extract_section_title(text):
    """Extracts the first meaningful title from a chunk to assign proper titles."""
    match = re.match(r"^(.*?) – ", text)  # Look for "TITLE – Content"
    return match.group(1).strip() if match else "Untitled Section"

def semantic_chunking(text, max_tokens=512):
    """Splits text into semantically meaningful chunks while keeping logical breaks."""
    tokenizer = get_tokenizer()
    paragraphs = split_into_paragraphs(text)
    chunks = []
    current_chunk = []
    current_length = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph, tokenizer)

        # If paragraph is too long, split further by sentences
        if paragraph_tokens > max_tokens:
            sentences = paragraph.split(". ")
            for sentence in sentences:
                sentence_tokens = count_tokens(sentence, tokenizer)

                if current_length + sentence_tokens > max_tokens:
                    chunks.append(" ".join(current_chunk))
                    current_chunk = []
                    current_length = 0

                current_chunk.append(sentence)
                current_length += sentence_tokens
        else:
            if current_length + paragraph_tokens > max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

            current_chunk.append(paragraph)
            current_length += paragraph_tokens

    if current_chunk:
        chunks.append(" ".join(current_chunk))  # Add the last chunk

    return chunks

def process_extracted(input_folder="data/processed", output_folder="data/chunked", max_tokens=512):
    """Loads extracted JSON files, applies semantic chunking, and saves chunked output."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".json"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            chunked_data = {
                "document_name": data.get("document_name", ""),
                "chunks": []
            }

            for section in data.get("sections", []):
                section_title = section.get("title", "Untitled Section")
                content = " ".join(section.get("content", []))

                section_chunks = semantic_chunking(content, max_tokens)
                for chunk in section_chunks:
                    chunked_data["chunks"].append({
                        "title": section_title if section_title != "list_item" else extract_section_title(chunk),
                        "text": chunk
                    })

            # Preserve tables & lists
            chunked_data["tables"] = data.get("tables", [])
            chunked_data["lists"] = data.get("lists", [])

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(chunked_data, f, indent=4, ensure_ascii=False)

            print(f" Chunked & Saved: {filename}")

    print("\n All documents processed and chunked! Output saved in 'data/chunked'.")
