import os
import glob
import numpy as np
import faiss
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Setup for RAG using FAISS and Sentence-Transformers
try:
    # We will use a more lightweight approach without Langchain's Chroma for FAISS
    HAS_RAG_DEPS = True
except ImportError:
    HAS_RAG_DEPS = False
    print("Warning: RAG dependencies not found. Please pip install faiss-cpu sentence-transformers")

INDEX_PATH = "./faiss_index.bin"
DOCS_PATH = "./docs_metadata.npy"

class RAGEngine:
    def __init__(self):
        self.enabled = HAS_RAG_DEPS
        if not self.enabled:
            return
            
        # Use a high-quality, local-friendly model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = [] # Stores the text chunks
        
        # Load index if it exists
        self.load_index()

    def load_index(self):
        if os.path.exists(INDEX_PATH) and os.path.exists(DOCS_PATH):
            try:
                self.index = faiss.read_index(INDEX_PATH)
                self.documents = np.load(DOCS_PATH, allow_pickle=True).tolist()
                print(f"Loaded RAG index with {len(self.documents)} chunks.")
            except Exception as e:
                print(f"Error loading index: {e}")
                self.index = None

    def save_index(self):
        if self.index is not None:
            faiss.write_index(self.index, INDEX_PATH)
            np.save(DOCS_PATH, np.array(self.documents, dtype=object))

    def chunk_text(self, text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
        """Simple text chunker."""
        chunks = []
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def process_file(self, file_path: str) -> List[str]:
        """Loads a file and returns text chunks."""
        if not self.enabled: return []
        
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext in ['.py', '.txt', '.md', '.css', '.js', '.html']:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return self.chunk_text(content)
            return []
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            return []

    def ingest_documents(self, file_paths: List[str]):
        """Ingests a list of files into the FAISS index."""
        if not self.enabled: return False
        
        all_chunks = []
        for path in file_paths:
            chunks = self.process_file(path)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            return False

        # Embed all chunks
        embeddings = self.model.encode(all_chunks)
        embeddings = np.array(embeddings).astype('float32')

        # Initialize or add to FAISS index
        dimension = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dimension)
        
        self.index.add(embeddings)
        self.documents.extend(all_chunks)
        
        self.save_index()
        return True
        
    def query(self, query_text: str, k: int = 3) -> str:
        """Queries the FAISS index and returns context."""
        if not self.enabled or self.index is None: return ""
        
        # Embed query
        query_vector = self.model.encode([query_text])
        query_vector = np.array(query_vector).astype('float32')
        
        # Search index
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                results.append(self.documents[idx])
        
        if not results: return ""
        
        context_str = "\n\n---\n\n".join(results)
        return context_str

    def reset_database(self):
        """Clears the FAISS index."""
        if os.path.exists(INDEX_PATH): os.remove(INDEX_PATH)
        if os.path.exists(DOCS_PATH): os.remove(DOCS_PATH)
        self.index = None
        self.documents = []

