from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import SQLDatabase
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.llms.base import LLM
from pydantic import Field
from typing import Optional, List, Mapping, Any
import requests
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)


def call_ollama(prompt, model_name="qwen3:14b", base_url="http://localhost:11434"):
    """Robust function to call Ollama API and handle NDJSON responses"""
    try:
        response = requests.post(
            f"{base_url}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False},
            stream=True,
            timeout=120  # Increased timeout for longer responses
        )
        response.raise_for_status()

        responses = []
        for line in response.iter_lines():
            if line:
                try:
                    data = json.loads(line.decode("utf-8"))
                    if "response" in data:
                        responses.append(data["response"])
                except json.JSONDecodeError:
                    continue

        return "".join(responses)

    except requests.exceptions.RequestException as e:
        logger.error(f"Error connecting to Ollama: {str(e)}")
        return f"Error connecting to Ollama: {str(e)}"


# 🔧 Custom LLM Class to connect with Ollama local server using call_ollama
class OllamaLLM(LLM):
    base_url: str = Field(default="http://localhost:11434")
    model_name: str = Field(default="gemma3:4b")

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # Use the robust call_ollama function here
        return call_ollama(prompt, model_name=self.model_name, base_url=self.base_url)

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"base_url": self.base_url, "model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        return "ollama-llm"


# 📂 Load text files from the directory
def load_text_documents(text_dir):
    loader = DirectoryLoader(text_dir, glob="*.txt", loader_cls=TextLoader)
    documents_from_text = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(documents_from_text)


# 🧾 Load world politics data from existing SQLite schema
def load_world_politics_data(db_url):
    db = SQLDatabase.from_uri(db_url)
    query = """
    SELECT pe.id, c.name as country, pe.event_name, pe.description, pe.date_event
    FROM political_events pe
    JOIN countries c ON pe.country_id = c.id
    """
    df = pd.read_sql(query, con=db._engine)

    documents = []
    for _, row in df.iterrows():
        content = (
            f"Country: {row['country']}\n"
            f"Event: {row['event_name']}\n"
            f"Description: {row['description']}\n"
            f"Date: {row['date_event']}"
        )
        documents.append(Document(page_content=content, metadata={"id": row["id"]}))

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    return text_splitter.split_documents(documents)


# 🧠 Build FAISS vector store with local embeddings
def build_faiss_index(texts):
    embedder = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    faiss_vectorstore = FAISS.from_documents(texts, embedding=embedder)
    return faiss_vectorstore


# 🤖 Create RAG pipeline with Ollama LLM
def create_rag_pipeline(vectorstore):
    llm = OllamaLLM(base_url="http://localhost:11434", model_name="gemma3:4b")
    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    return rag_chain


# 🧪 Run a test query
def run_test_query(rag_pipeline):
    query = "Tell me about recent political events in the world."
    response = rag_pipeline({"query": query})
    print("Answer:", response["result"])
    print("\nSource Documents:")
    for doc in response["source_documents"]:
        print(f"- {doc.page_content}\n")


# Helper to format SQLite URL correctly
def format_sqlite_url(path):
    if not path.startswith("sqlite:///"):
        if path.startswith("/"):
            return f"sqlite:///{path}"
        else:
            return f"sqlite:///{path}"
    return path


# 🚀 Main execution
if __name__ == "__main__":
    # Hardcoded paths
    text_dir = "/home/harsha/Dev/indiheritage/data/religious"
    sqlite_path = format_sqlite_url("/home/harsha/Dev/indiheritage/data/religious/world_politics.db")

    # Load documents from text files
    docs = load_text_documents(text_dir)

    # Load documents from world politics database
    docs.extend(load_world_politics_data(sqlite_path))

    # Build FAISS index
    vectorstore = build_faiss_index(docs)

    # Create RAG pipeline
    rag_pipeline = create_rag_pipeline(vectorstore)

    # Run test query
    run_test_query(rag_pipeline)
