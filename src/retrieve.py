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


import warnings

# suppress all wanings
warnings.filterwarnings("ignore")

# Load the Sample FAQ Document and Split into Chunks
loader = DirectoryLoader(path="./data", recursive=True, glob="*.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

docs = text_splitter.split_documents(documents)

# Convert Text to Vectors and Index with FAISS
vector_store = FAISS.from_documents(docs, embeddings)

# Initialize a retriever for querying the vector store
retriever = vector_store.as_retriever(search_type="similarity", search_k=3)

# Setup Retrieval-Augmented QA Chain
retrieval_qa_chain  = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# system_prompt = (
#     "Use the given context to answer the question. "
#     "If you don't know the answer, say you don't know. "
#     "Use three sentence maximum and keep the answer concise. "
#     "Context: {context}"
# )
# prompt = ChatPromptTemplate.from_messages(
#     [
#         ("system", system_prompt),
#         ("human", "{input}"),
#     ]
# )


# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# chain = create_retrieval_chain(retriever, question_answer_chain)
