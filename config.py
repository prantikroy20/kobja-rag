import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # OpenAI Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # Model Configuration
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Sentence transformer model
    LLM_MODEL = "gpt-3.5-turbo"  # Can be changed to gpt-4 or other models
    
    # RAG Configuration
    TOP_K_QUESTIONS = 5  # Number of similar questions to retrieve
    TOP_K_TEXTBOOK = 3   # Number of relevant textbook chunks to retrieve
    CHUNK_SIZE = 500     # Size of textbook chunks
    CHUNK_OVERLAP = 50   # Overlap between chunks
    
    # Vector Database
    VECTOR_DB_PATH = "./vector_db"
    COLLECTION_NAME_QUESTIONS = "questions_collection"
    COLLECTION_NAME_TEXTBOOK = "textbook_collection"
    
    # Generation Parameters
    MAX_TOKENS = 500
    TEMPERATURE = 0.7