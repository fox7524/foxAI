import os
import glob
from typing import List

# Setup for Langchain & Chroma
try:
    from langchain.docstore.document import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.document_loaders import (
        PyMuPDFLoader,
        Docx2txtLoader,
        TextLoader,
        BSHTMLLoader
    )
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False
    print("Warning: RAG dependencies not found. Please pip install langchain chromadb sentence-transformers pymupdf python-docx")

DB_DIR = "./chroma_db"

class RAGEngine:
    def __init__(self):
        self.enabled = HAS_RAG_DEPS
        if not self.enabled:
            return
            
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        
        # Initialize chroma db
        self.db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)

    def process_file(self, file_path: str) -> List['Document']:
        """Loads a file based on its extension and returns splitted documents."""
        if not self.enabled: return []
        
        ext = os.path.splitext(file_path)[1].lower()
        documents = []
        
        try:
            if ext == '.pdf':
                loader = PyMuPDFLoader(file_path)
                documents = loader.load()
            elif ext in ['.docx', '.doc']:
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
            elif ext in ['.html', '.htm']:
                loader = BSHTMLLoader(file_path)
                documents = loader.load()
            elif ext in ['.py', '.cpp', '.c', '.h', '.ino', '.js', '.css', '.txt', '.md']:
                # Generic text loader for code and text
                loader = TextLoader(file_path, encoding='utf-8')
                documents = loader.load()
            elif ext == '.zim':
                # ZIM files are complex. A simplistic mock-up approach or warning.
                documents = [Document(page_content=f"[ZIM File Detected: {file_path}. Native parsing requires specialized library. Logging meta only.]", metadata={"source": file_path})]
            else:
                # Fallback to pure text
                loader = TextLoader(file_path, autodetect_encoding=True)
                documents = loader.load()
                
            return self.text_splitter.split_documents(documents)
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def ingest_documents(self, file_paths: List[str]):
        """Ingests a list of files into the vector database."""
        if not self.enabled: return False
        
        all_splits = []
        for path in file_paths:
            splits = self.process_file(path)
            all_splits.extend(splits)
            
        if all_splits:
            self.db.add_documents(all_splits)
            return True
        return False
        
    def query(self, query_text: str, k: int = 3) -> str:
        """Queries the vector database and formats context."""
        if not self.enabled: return ""
        
        results = self.db.similarity_search(query_text, k=k)
        if not results: return ""
        
        context_str = "\n\n---\n\n".join([r.page_content for r in results])
        return context_str

    def reset_database(self):
        """Clears the persistent chroma directory."""
        if not self.enabled: return
        self.db.delete_collection()
        self.db = Chroma(persist_directory=DB_DIR, embedding_function=self.embeddings)
