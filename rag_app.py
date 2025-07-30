import streamlit as st
import json
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document

# Load OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Config
INDEX_DIR = "chroma_index"  # Folder containing chroma.sqlite3 and other .bin files
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

# Load Chroma vector store
@st.cache_resource
def load_vector_store():
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=openai_api_key)

    if os.path.exists(os.path.join(INDEX_DIR, "chroma.sqlite3")):
        return Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings)
    else:
        st.error(f"Chroma vector store not found in {INDEX_DIR}")
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

# Initialize chat history
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# UI Layout
st.set_page_config(page_title="UChicago MS-ADS RAG Chatbot", layout="wide")
st.title("🎓 MS in Applied Data Science Chatbot")
st.markdown("Ask a question about the program, curriculum, or admissions.")

query = st.text_input("Enter your question:").strip()

if query:
    with st.spinner("Generating answer..."):
        chain = load_qa_chain()
        answer = chain.run(query)

        st.session_state.qa_history.append((query, answer))

# Display history
if st.session_state.qa_history:
    st.subheader("💬 Chat History")
    total = len(st.session_state.qa_history)
    for i, (q, a) in enumerate(reversed(st.session_state.qa_history), 1):
        label_num = total - i + 1
        st.markdown(f"**Q{label_num}: {q}**")
        st.markdown(f"**A{label_num}:** {a}")