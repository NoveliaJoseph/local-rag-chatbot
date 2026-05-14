from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import pdfplumber
import faiss
import numpy as np
import ollama
import os
import json

# EMBEDDING MODEL
model = SentenceTransformer('all-MiniLM-L6-v2')

# PDF EXTRACTION
def extract_pdf_text(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text

# TEXT CHUNKING
def create_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )
    chunks = text_splitter.split_text(text)
    return chunks

# CREATE EMBEDDINGS
def create_embeddings(chunks):
    embeddings = model.encode(
        chunks,
        show_progress_bar=True
    )
    return embeddings

# STORE EMBEDDINGS IN FAISS
def store_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(
        np.array(embeddings).astype("float32")
    )
    return index

# SAVE VECTORSTORE
def save_vectorstore(index, chunks):
    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, "vectorstore/faiss_index.index")
    with open("vectorstore/chunks.json", "w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=4)

# LOAD VECTORSTORE
def load_vectorstore():
    index = faiss.read_index("vectorstore/faiss_index.index")
    with open("vectorstore/chunks.json", "r", encoding="utf-8") as file:
        chunks = json.load(file)
    return index, chunks

# INITIALIZE RAG
def initialize_rag(pdf_path, session_id):

    vectorstore_dir = f"media/vectorstores/{session_id}"

    os.makedirs(vectorstore_dir, exist_ok=True)

    faiss_path = f"{vectorstore_dir}/faiss.index"
    chunks_path = f"{vectorstore_dir}/chunks.json"

    print("Creating new vectorstore...")

    pdf_text = extract_pdf_text(pdf_path)

    if not pdf_text.strip():
        raise ValueError("No readable text found in PDF. It might be scanned or empty.")

    chunks = create_chunks(pdf_text)

    embeddings = create_embeddings(chunks)

    index = store_in_faiss(embeddings)

    faiss.write_index(index, faiss_path)

    with open(chunks_path, "w", encoding="utf-8") as file:
        json.dump(chunks, file)

    return index, chunks

# SEARCH QUERY
def search_query(question, model, index, chunks):
    question_embedding = model.encode([question])
    distances, indices = index.search(
        np.array(question_embedding).astype("float32"),
        k=2
    )
    retrieved_chunks = []
    for i in range(len(indices[0])):
        chunk_index = indices[0][i]
        retrieved_chunks.append(chunks[chunk_index])
    return "\n\n".join(retrieved_chunks)

# LOCAL LLM RESPONSE
def ask_llm(question, context):
    if not context:
        return "I cannot find this information in the document"

    prompt = f"""
You are a professional AI document assistant.
You MUST answer ONLY from the provided document context.

STRICT RULES:
- Do NOT use outside knowledge
- Do NOT guess
- If answer is unavailable in context, say EXACTLY:
"I cannot find this information in the document"

ANSWER STYLE:
- Use headings
- Use bullet points when useful
- Keep answers clean and readable
- Format like ChatGPT
- Use code blocks for code
- Explain clearly but briefly

DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

FINAL ANSWER:
"""
    response = ollama.chat(
        model="phi3:latest",
        messages=[{"role": "user", "content": prompt}]
    )
    return response["message"]["content"]