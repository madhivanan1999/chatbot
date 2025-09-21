import streamlit as st
from langchain_ollama import ChatOllama
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import os

def mod(model='llama3.2:1b'):
    llm = ChatOllama(
                # base_url='http://127.0.0.1:11434',
                model=model,
                temperature=0.5,
            )
    return llm

 
# Initialize ChromaDB
DB_DIR = "chroma_db"
os.makedirs(DB_DIR, exist_ok=True)

# Initialize FAISS
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = 384  # all-MiniLM-L6-v2 embedding size
index = faiss.IndexFlatL2(dimension)

# Store metadata (text chunks)
documents = []
texts =[]
 
# Streamlit UI
st.set_page_config(page_title="Ollama RAG Chatbot - AI", page_icon="📚")
st.title("📖 Ollama RAG Chatbot")
 
# Sidebar - Model Selection
with st.sidebar:
    st.header("🔍 Settings")
    selected_model = st.selectbox("Choose Ollama Model:", ["llama3.2:1b"])
    st.write(f"Using `{selected_model}` model")
    llms = mod(selected_model)
 
# File Upload
uploaded_file = st.file_uploader("📂 Upload a PDF to process", type=['pdf'])
if uploaded_file:
    st.success("PDF Uploaded Successfully!")
    
    temp_file = f"./{uploaded_file.name}"
    with open(temp_file, "wb") as file:
       file.write(uploaded_file.getvalue())
       file_name = uploaded_file.name
    # Process PDF
    if ".pdf" in temp_file:
        loader = PyPDFLoader(temp_file)
        documents = loader.load()
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)
 

     # Convert text chunks to embeddings and store in FAISS
    texts = [chunk.page_content for chunk in chunks]
    embeddings = embedding_model.encode(texts)
    index.add(np.array(embeddings))

    # Store in ChromaDB
    # FAISS.from_documents(chunks,embedding_model)
    st.success("📚 Document added to knowledge base!")
 
# Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []
 
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
 
# User Input
user_query = st.chat_input("Type your question...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    if len(texts) >= 1:
        # Convert query to embedding and search FAISS
        query_embedding = embedding_model.encode([user_query])
        _, indices = index.search(np.array(query_embedding), k=1)

        context = "\n".join([texts[i] for i in indices[0]])     
    
        # Generate Response using Ollama
        prompt = f"Context: {context}\n\nUser Query: {user_query}\n\nAnswer:"
        # response = ollama.chat(model=selected_model, messages=[{"role": "user", "content": prompt}])
    else:
        prompt = user_query
  
    
    response = llms.invoke([{"role": "user", "content": prompt}])
    bot_reply = response.content
 
    # Display Response
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)