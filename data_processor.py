import pandas as pd
import re
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DataProcessor:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s\.\?\!\,\;\:\-\(\)]', '', text)
        return text.strip()
    
    def process_questions(self, questions_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process old questions data
        Expected format: [{"question": "...", "answer": "...", "topic": "...", "difficulty": "..."}]
        """
        processed_questions = []
        
        for i, item in enumerate(questions_data):
            question = self.clean_text(item.get("question", ""))
            answer = self.clean_text(item.get("answer", ""))
            topic = item.get("topic", "General")
            difficulty = item.get("difficulty", "Medium")
            
            if question:  # Only process if question exists
                processed_questions.append({
                    "id": f"q_{i}",
                    "question": question,
                    "answer": answer,
                    "topic": topic,
                    "difficulty": difficulty,
                    "content": f"Question: {question}\nAnswer: {answer}\nTopic: {topic}"
                })
        
        return processed_questions
    
    def process_textbook(self, textbook_content: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process textbook content into chunks
        """
        if metadata is None:
            metadata = {}
        
        # Clean the textbook content
        cleaned_content = self.clean_text(textbook_content)
        
        # Split into chunks
        chunks = self.text_splitter.split_text(cleaned_content)
        
        processed_chunks = []
        for i, chunk in enumerate(chunks):
            processed_chunks.append({
                "id": f"tb_{i}",
                "content": chunk,
                "chapter": metadata.get("chapter", "Unknown"),
                "subject": metadata.get("subject", "General"),
                "page": metadata.get("page", i),
                "metadata": metadata
            })
        
        return processed_chunks
    
    def load_questions_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load questions from various file formats"""
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            return df.to_dict('records')
        elif file_path.endswith('.json'):
            import json
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError("Unsupported file format. Use CSV or JSON.")
    
    def load_textbook_from_file(self, file_path: str) -> str:
        """Load textbook content from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()