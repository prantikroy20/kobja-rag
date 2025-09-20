import json
from question_generator import QuestionGenerator
from config import Config

def main():
    # Initialize configuration
    config = Config()
    
    # Create question generator
    generator = QuestionGenerator(config)
    
    # Sample data for demonstration
    sample_questions = [
        {
            "question": "What is the capital of France?",
            "answer": "Paris",
            "topic": "Geography",
            "difficulty": "Easy"
        },
        {
            "question": "Explain the process of photosynthesis in plants.",
            "answer": "Photosynthesis is the process by which plants convert light energy into chemical energy...",
            "topic": "Biology",
            "difficulty": "Medium"
        },
        {
            "question": "What is the derivative of x^2 + 3x + 2?",
            "answer": "2x + 3",
            "topic": "Mathematics",
            "difficulty": "Medium"
        }
    ]
    
    sample_textbook = """
    Chapter 1: Introduction to Biology
    
    Biology is the scientific study of life and living organisms. It encompasses various fields
    including molecular biology, genetics, ecology, and evolution. Living organisms share
    several characteristics: they are composed of cells, maintain homeostasis, grow and
    reproduce, respond to their environment, and evolve over time.
    
    Photosynthesis is a crucial biological process that occurs in plants, algae, and some
    bacteria. During photosynthesis, these organisms convert light energy, usually from
    the sun, into chemical energy stored in glucose molecules. The general equation for
    photosynthesis is: 6CO2 + 6H2O + light energy → C6H12O6 + 6O2.
    
    Chapter 2: Cell Structure and Function
    
    Cells are the basic units of life. There are two main types of cells: prokaryotic and
    eukaryotic. Prokaryotic cells lack a membrane-bound nucleus, while eukaryotic cells
    have a nucleus enclosed by a nuclear membrane. Plant cells have additional structures
    like chloroplasts and cell walls that are not found in animal cells.
    """
    
    print("Initializing Question Generator...")
    
    # Initialize database with sample data
    generator.initialize_database(
        questions_data=sample_questions,
        textbook_content=sample_textbook
    )
    
    # Get database statistics
    stats = generator.get_database_stats()
    print(f"Database Stats: {json.dumps(stats, indent=2)}")
    
    # Generate a new question
    print("\nGenerating new question...")
    new_question = generator.generate_new_question(
        topic="Biology",
        difficulty="Medium",
        question_type="Multiple Choice",
        top_k_questions=3,
        top_k_textbook=2
    )
    
    print(f"Generated Question: {json.dumps(new_question, indent=2)}")
    
    # Generate multiple questions
    print("\nGenerating multiple questions...")
    topics = ["Biology", "Mathematics"]
    batch_questions = generator.batch_generate_questions(
        topics=topics,
        difficulty="Medium",
        questions_per_topic=2
    )
    
    print(f"Generated {len(batch_questions)} questions")
    
    # Evaluate questions
    print("\nEvaluating generated questions...")
    evaluated_questions = generator.evaluate_generated_questions(batch_questions)
    
    for i, q in enumerate(evaluated_questions):
        print(f"\nQuestion {i+1}:")
        print(f"Topic: {q.get('topic', 'N/A')}")
        print(f"Question: {q.get('question', 'N/A')}")
        if 'evaluation' in q and 'overall' in q['evaluation']:
            print(f"Quality Score: {q['evaluation']['overall']}/10")

if __name__ == "__main__":
    main()