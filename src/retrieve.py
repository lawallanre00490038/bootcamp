from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import DirectoryLoader

from dotenv import load_dotenv
import os

from model import llm, embeddings



# Optional: Avoid timeout issues with Hugging Face downloads
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "60"

# === Load & Prepare Documents ===
print("Loading documents...")
loader = DirectoryLoader(path="./data", recursive=True, glob="*.pdf")
documents = loader.load()

print(f"Loaded {len(documents)} document(s). Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# === Build Vector Store ===
print("Building FAISS vector store...")
vector_store = FAISS.from_documents(docs, embeddings)
retriever = vector_store.as_retriever(search_type="similarity", search_k=3)

# Setup Retrieval-Augmented QA Chain
retrieval_qa_chain  = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)
