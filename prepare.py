import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


pdf_path = 'data/RAG.pdf'
loader = PyMuPDFLoader(pdf_path)
documents = loader.load()

print(f"Loaded {len(documents)} pages from the PDF")
print(f"Sample text from first page: {documents[0].page_content[:200]}...")


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=10,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = text_splitter.split_documents(documents)
print(f"Split document into {len(chunks)} chunks")
print(f"Sample chunk: {chunks[0].page_content[:150]}...")

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Using local HuggingFace embedding model: sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(chunks, embedding_model)
print("Vector database created successfully")

vectorstore.save_local("rag_survey_faiss_index")
print("Vector database saved locally for future use")

