import faiss #vector db #vector search engine
import os
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup #html tag removal
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from groq import Groq
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("Groq_api_key")
Groq_model = Groq(api_key= api_key)

def pdf(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    text = []
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
        doc = loader.load()
        for page in doc:
            text.append(page.page_content)
    elif ext == ".docx":
        doc = Document(file_path)
        for para in doc.paragraphs:
            text.append(para.text)
    else:
        print("Unsupported file format")

    return "\n".join(text)

def extract_table(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".xlsx", ".xls", ".xlsm"]:
        df = pd.read_excel(file_path)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        print("unsupported file format")
    return df.to_string()

def url_link(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

def cleaning(text):
    return text.replace("\n"," ").strip()

def chunking(text):
    splitter = RecursiveCharacterTextSplitter( chunk_size = 500, chunk_overlap = 50)
    return splitter.split_text(text)

#embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = SentenceTransformer("multi-qa-MiniLM-L6-cos-v1")

def embed_chunks(chunks):
    embeddings = embedding_model.encode(chunks)
    return np.array(embeddings).astype("float32")

def faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) #L2 (Euclidean) ,Cosine similarity, Inner product ## Compare the query vector with every stored vector, compute distances, sort them, and return the top K closest ones
    index.add(embeddings)
    return index
            
def retrieve_chunks(query, chunks, index, k=3):
    query_embedding = embedding_model.encode([query]) #Because embedding model expects list of sentences
    query_embedding = np.array(query_embedding).astype("float32")
    distances, indices = index.search(query_embedding, k)
    return [chunks[i] for i in indices[0]] 

def build_prompt(context_chunks, question):
    context = "\n\n".join(context_chunks)

    return f"""
You are a helpful assistant.
Answer ONLY using the context below.
If answer is not found, say "I don't know".

Context:
{context}

Question:
{question}
"""
def ask_groq(prompt):
    response = Groq_model.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    return response.choices[0].message.content

text = extract_table(r"C:\Users\MADHIVANAN\OneDrive\Desktop\dummy_data_for_testing.csv")# Pdf # or extract_table() or url_link()
text = cleaning(text)

chunks = chunking(text)
embeddings = embed_chunks(chunks)
index = faiss_index(embeddings)

print("\n Document loaded. Ask questions about it!")
print("Type 'exit' to quit.\n")

while True:
    question = input("Enter your Query: ").strip()

    if question.lower() == "exit":
        print("Session ended ----- Bye!")
        break

    relevant_chunks = retrieve_chunks(question, chunks, index)
    prompt = build_prompt(relevant_chunks, question)
    answer = ask_groq(prompt)

    print("\n Answer:")
    print(answer)
    print("-" * 60)

