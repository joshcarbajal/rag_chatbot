import os
os.environ["CHROMA_DB_IMPL"] = "duckdb"  # ✅ Must be set before any Chroma import

import streamlit as st
import json
import sqlite3
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma

# Config
INDEX_DIR = "chroma_index"
DATA_FILE = "scraped_ms_ads_data_v3.json"
DB_FILE = "qa_history.db"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

# Load OpenAI API Key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Set up SQLite DB
conn = sqlite3.connect(DB_FILE)
cursor = conn.cursor()
cursor.execute("""
    CREATE TABLE IF NOT EXISTS history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        question TEXT,
        answer TEXT
    )
""")
conn.commit()

# Load or build Chroma vector store
@st.cache_resource
def load_vector_store():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=openai_api_key)

    metadata_path = os.path.join(INDEX_DIR, "collection_metadata.json")
    db_path = os.path.join(INDEX_DIR, "chroma.sqlite3")

    if os.path.exists(metadata_path) and os.path.exists(db_path):
        return Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)

    elif os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        documents = [
            Document(page_content=item.get("content", ""), metadata={k: v for k, v in item.items() if k != "content"})
            for item in raw_data
        ]

        splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
        split_docs = splitter.split_documents(documents)

        vectorstore = Chroma.from_documents(
            documents=split_docs,
            embedding=embeddings,
            persist_directory=INDEX_DIR
        )
        vectorstore.persist()
        return vectorstore

    else:
        st.error("❌ Chroma index not found and JSON data unavailable to rebuild.")
        st.stop()

# Load QA chain
@st.cache_resource
def load_qa_chain():
    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatOpenAI(
        temperature=0.0,
        model_name=LLM_MODEL,
        openai_api_key=openai_api_key
    )

    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)

# Streamlit app
st.set_page_config(page_title="MS-ADS RAG Chatbot", layout="wide")
st.title("🎓 MS in Applied Data Science Chatbot")

query = st.text_input("Enter your question:").strip()

if query:
    with st.spinner("Generating answer..."):
        chain = load_qa_chain()
        answer = chain.run(query)

        # Store in SQLite
        cursor.execute("INSERT INTO history (question, answer) VALUES (?, ?)", (query, answer))
        conn.commit()

        st.markdown(f"**Q: {query}**")
        st.markdown(f"**A:** {answer}")

# Display previous Q&A
if st.checkbox("Show chat history"):
    st.subheader("💬 Chat History")
    cursor.execute("SELECT question, answer FROM history ORDER BY id DESC")
    rows = cursor.fetchall()
    for i, (q, a) in enumerate(rows, 1):
        st.markdown(f"**Q{i}: {q}**")
        st.markdown(f"**A{i}:** {a}")
