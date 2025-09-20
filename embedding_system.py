from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import uuid

class CustomEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input, convert_to_tensor=False)
        return embeddings.tolist()

class EmbeddingSystem:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", db_path: str = "./vector_db"):
        self.model = SentenceTransformer(model_name)
        self.embedding_function = CustomEmbeddingFunction(model_name)
        self.client = chromadb.PersistentClient(path=db_path)
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for a list of texts"""
        embeddings = self.model.encode(texts, convert_to_tensor=False)
        return embeddings
    
    def setup_collections(self, questions_collection_name: str, textbook_collection_name: str):
        """Set up ChromaDB collections for questions and textbook content"""
        # Create or get collections with custom embedding function
        self.questions_collection = self.client.get_or_create_collection(
            name=questions_collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.textbook_collection = self.client.get_or_create_collection(
            name=textbook_collection_name,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_questions_to_db(self, questions: List[Dict[str, Any]]):
        """Add processed questions to the vector database"""
        texts = [q["content"] for q in questions]
        
        ids = [q["id"] for q in questions]
        metadatas = [{
            "topic": q["topic"],
            "difficulty": q["difficulty"],
            "question": q["question"],
            "answer": q["answer"]
        } for q in questions]
        
        # ChromaDB will use our custom embedding function automatically
        self.questions_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def add_textbook_to_db(self, textbook_chunks: List[Dict[str, Any]]):
        """Add processed textbook chunks to the vector database"""
        texts = [chunk["content"] for chunk in textbook_chunks]
        
        ids = [chunk["id"] for chunk in textbook_chunks]
        metadatas = [{
            "chapter": chunk["chapter"],
            "subject": chunk["subject"],
            "page": chunk["page"]
        } for chunk in textbook_chunks]
        
        # ChromaDB will use our custom embedding function automatically
        self.textbook_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def search_similar_questions(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar questions using semantic similarity"""
        query_embedding = self.create_embeddings([query])
        
        results = self.questions_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        similar_questions = []
        for i in range(len(results['ids'][0])):
            similar_questions.append({
                "id": results['ids'][0][i],
                "question": results['metadatas'][0][i]['question'],
                "answer": results['metadatas'][0][i]['answer'],
                "topic": results['metadatas'][0][i]['topic'],
                "difficulty": results['metadatas'][0][i]['difficulty'],
                "similarity_score": 1 - results['distances'][0][i],  # Convert distance to similarity
                "content": results['documents'][0][i]
            })
        
        return similar_questions
    
    def search_relevant_textbook(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for relevant textbook content"""
        query_embedding = self.create_embeddings([query])
        
        results = self.textbook_collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=top_k
        )
        
        relevant_content = []
        for i in range(len(results['ids'][0])):
            relevant_content.append({
                "id": results['ids'][0][i],
                "content": results['documents'][0][i],
                "chapter": results['metadatas'][0][i]['chapter'],
                "subject": results['metadatas'][0][i]['subject'],
                "page": results['metadatas'][0][i]['page'],
                "similarity_score": 1 - results['distances'][0][i]
            })
        
        return relevant_content