from typing import List, Dict, Any, Optional
from data_processor import DataProcessor
from embedding_system import EmbeddingSystem
from llm_integration import LLMIntegration
from config import Config

class QuestionGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.data_processor = DataProcessor(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.embedding_system = EmbeddingSystem(
            model_name=config.EMBEDDING_MODEL,
            db_path=config.VECTOR_DB_PATH
        )
        self.llm = LLMIntegration(
            api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL
        )
        
        # Setup collections
        self.embedding_system.setup_collections(
            config.COLLECTION_NAME_QUESTIONS,
            config.COLLECTION_NAME_TEXTBOOK
        )
    
    def initialize_database(self, 
                          questions_file: str = None,
                          textbook_file: str = None,
                          questions_data: List[Dict[str, Any]] = None,
                          textbook_content: str = None):
        """Initialize the vector database with questions and textbook content"""
        
        # Process and add questions
        if questions_file:
            questions_data = self.data_processor.load_questions_from_file(questions_file)
        
        if questions_data:
            processed_questions = self.data_processor.process_questions(questions_data)
            self.embedding_system.add_questions_to_db(processed_questions)
            print(f"Added {len(processed_questions)} questions to database")
        
        # Process and add textbook content
        if textbook_file:
            textbook_content = self.data_processor.load_textbook_from_file(textbook_file)
        
        if textbook_content:
            textbook_chunks = self.data_processor.process_textbook(textbook_content)
            self.embedding_system.add_textbook_to_db(textbook_chunks)
            print(f"Added {len(textbook_chunks)} textbook chunks to database")
    
    def generate_new_question(self,
                            topic: str,
                            difficulty: str = "Medium",
                            question_type: str = "Multiple Choice",
                            top_k_questions: int = None,
                            top_k_textbook: int = None) -> Dict[str, Any]:
        """Generate a new question based on topic using RAG approach"""
        
        if top_k_questions is None:
            top_k_questions = self.config.TOP_K_QUESTIONS
        if top_k_textbook is None:
            top_k_textbook = self.config.TOP_K_TEXTBOOK
        
        # Step 1: Retrieve similar questions using RAG
        similar_questions = self.embedding_system.search_similar_questions(
            query=f"{topic} {difficulty} {question_type}",
            top_k=top_k_questions
        )
        
        # Step 2: Retrieve relevant textbook content
        relevant_textbook = self.embedding_system.search_relevant_textbook(
            query=topic,
            top_k=top_k_textbook
        )
        
        # Step 3: Generate new question using LLM
        generated_question = self.llm.generate_question(
            topic=topic,
            similar_questions=similar_questions,
            textbook_content=relevant_textbook,
            difficulty=difficulty,
            question_type=question_type,
            max_tokens=self.config.MAX_TOKENS,
            temperature=self.config.TEMPERATURE
        )
        
        # Step 4: Add metadata about the generation process
        generated_question.update({
            "generation_metadata": {
                "similar_questions_count": len(similar_questions),
                "textbook_chunks_used": len(relevant_textbook),
                "top_k_questions": top_k_questions,
                "top_k_textbook": top_k_textbook,
                "similar_questions": [q["question"] for q in similar_questions[:3]],
                "similarity_scores": [q["similarity_score"] for q in similar_questions[:3]]
            }
        })
        
        return generated_question
    
    def batch_generate_questions(self,
                               topics: List[str],
                               difficulty: str = "Medium",
                               question_type: str = "Multiple Choice",
                               questions_per_topic: int = 1) -> List[Dict[str, Any]]:
        """Generate multiple questions for multiple topics"""
        
        all_questions = []
        
        for topic in topics:
            print(f"Generating {questions_per_topic} questions for topic: {topic}")
            
            for i in range(questions_per_topic):
                question = self.generate_new_question(
                    topic=topic,
                    difficulty=difficulty,
                    question_type=question_type
                )
                question["batch_id"] = f"{topic}_{i+1}"
                all_questions.append(question)
        
        return all_questions
    
    def evaluate_generated_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Evaluate a batch of generated questions"""
        
        evaluated_questions = []
        
        for question in questions:
            evaluation = self.llm.evaluate_question_quality(question)
            question["evaluation"] = evaluation
            evaluated_questions.append(question)
        
        return evaluated_questions
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the current database"""
        try:
            questions_count = self.embedding_system.questions_collection.count()
            textbook_count = self.embedding_system.textbook_collection.count()
            
            return {
                "questions_in_database": questions_count,
                "textbook_chunks_in_database": textbook_count,
                "database_path": self.config.VECTOR_DB_PATH
            }
        except Exception as e:
            return {"error": f"Failed to get database stats: {str(e)}"}