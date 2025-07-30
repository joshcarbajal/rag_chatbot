from flask import Flask, request, jsonify
import json
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb.config import Settings

# Flask app
app = Flask(__name__)

# Config
INDEX_DIR = "chroma_index"
DATA_FILE = "scraped_ms_ads_data_v3.json"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo"

# Load OpenAI API Key from environment variable
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Missing OPENAI_API_KEY environment variable")

# Load or build Chroma vector store
vectorstore = None

def load_vector_store():
    global vectorstore
    if vectorstore:
        return vectorstore

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, openai_api_key=openai_api_key)
    chroma_settings = Settings(chroma_db_impl="duckdb", persist_directory=INDEX_DIR)

    if os.path.exists(os.path.join(INDEX_DIR, "chroma.sqlite3")):
        vectorstore = Chroma(persist_directory=INDEX_DIR, embedding_function=embeddings, client_settings=chroma_settings)
        return vectorstore

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
            persist_directory=INDEX_DIR,
            client_settings=chroma_settings
        )
        vectorstore.persist()
        return vectorstore

    else:
        raise RuntimeError("Neither Chroma index nor JSON data found to build the vector store.")

# Load QA chain
qa_chain = None

def load_qa_chain():
    global qa_chain
    if qa_chain:
        return qa_chain

    vectorstore = load_vector_store()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatOpenAI(
        temperature=0.0,
        model_name=LLM_MODEL,
        openai_api_key=openai_api_key
    )

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return qa_chain

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("question", "").strip()

    if not query:
        return jsonify({"error": "No question provided."}), 400

    try:
        chain = load_qa_chain()
        answer = chain.run(query)
        return jsonify({"question": query, "answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/")
def index():
    return "MS-ADS RAG Chatbot is running. Use the /ask endpoint with POST JSON {\"question\": \"...\"}"
