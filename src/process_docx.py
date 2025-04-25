

import os
import json
from docling.document_converter import DocumentConverter

def extract_text_from_docling(doc):
    """Extracts all text sections from a Docling JSON document and assigns proper labels."""
    structured_sections = []
    current_section = None

    if "texts" in doc and isinstance(doc["texts"], list):
        for text_entry in doc["texts"]:
            if isinstance(text_entry, dict) and "text" in text_entry and text_entry["text"].strip():
                title = text_entry.get("label", "").strip()
                content = text_entry["text"].strip()

                #  Assign Proper Labels to Section Titles
                if title.lower() in ["scope", "purpose", "applies to", "background", "statement of policy"]:
                    section_type = "policy_section"
                elif "procedure" in title.lower():
                    section_type = "procedure_step"
                else:
                    section_type = "general_section"

                #  Ensure Sections Are Properly Grouped
                if not current_section or section_type != current_section["section_type"]:
                    if current_section:
                        structured_sections.append(current_section)
                    current_section = {
                        "section_type": section_type,
                        "title": title,
                        "content": []
                    }

                current_section["content"].append(content)

        if current_section:
            structured_sections.append(current_section)

    return structured_sections

def extract_tables_from_docling(doc):
    """Extracts tables from a Docling JSON document in a structured format."""
    structured_tables = []

    if "tables" in doc and isinstance(doc["tables"], list):
        for table_entry in doc["tables"]:
            if isinstance(table_entry, dict) and "data" in table_entry and "table_cells" in table_entry["data"]:
                headers = []
                rows = []
                for cell in table_entry["data"]["table_cells"]:
                    if isinstance(cell, dict) and "text" in cell and cell["text"].strip():
                        if cell.get("column_header", False):
                            headers.append(cell["text"].strip())
                        else:
                            rows.append(cell["text"].strip())

                # Table Headers & Row Structure
                if headers and rows:
                    structured_tables.append({
                        "headers": headers,
                        "rows": [rows[i:i+len(headers)] for i in range(0, len(rows), len(headers))]
                    })

    return structured_tables

def extract_lists_from_docling(doc):
    """Extracts bulleted and numbered lists from a Docling JSON document."""
    structured_lists = []

    if "lists" in doc and isinstance(doc["lists"], list):
        for list_entry in doc["lists"]:
            if isinstance(list_entry, dict) and "items" in list_entry:
                list_type = "ordered" if list_entry.get("list_type", "") == "ordered" else "unordered"
                list_items = [item["text"].strip() for item in list_entry["items"] if "text" in item and item["text"].strip()]
                
                if list_items:
                    structured_lists.append({
                        "list_type": list_type,
                        "items": list_items
                    })

    return structured_lists

def process_docx_with_docling(doc_path):
    """Processes a DOCX file using Docling and returns structured data."""
    converter = DocumentConverter()
    result = converter.convert(doc_path)
    doc = result.document.export_to_dict()

    structured_data = {
        "document_name": doc.get("name", os.path.basename(doc_path)),
        "sections": extract_text_from_docling(doc),
        "tables": extract_tables_from_docling(doc),
        "lists": extract_lists_from_docling(doc)
    }

    return structured_data

def process_all_documents(input_folder="data/raw", output_folder="data/processed"):
    """Processes all DOCX files and saves structured JSON output for LLM training."""
    os.makedirs(output_folder, exist_ok=True)

    for filename in os.listdir(input_folder):
        if filename.endswith(".docx"):
            doc_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename.replace(".docx", ".json"))

            extracted_data = process_docx_with_docling(doc_path)

            with open(output_path, "w", encoding="utf-8") as json_file:
                json.dump(extracted_data, json_file, indent=4, ensure_ascii=False)

            print(f" Processed: {filename} → {output_path}")
