import streamlit as st
from question_generator import QuestionGenerator
from config import Config
import json
import os

# Initialize session state
if 'generator' not in st.session_state:
    st.session_state.generator = None
    st.session_state.data_loaded = False

st.title("🤖 RAG-Based Question Generator")
st.markdown("Generate new questions using existing questions and textbook content with RAG + LLM")

# Sidebar for configuration
st.sidebar.header("Configuration")

# File uploads
st.sidebar.subheader("Data Sources")
questions_file = st.sidebar.file_uploader("Upload Existing Questions", type=['csv', 'json', 'txt'])
textbook_file = st.sidebar.file_uploader("Upload Textbook Content", type=['txt'])

# Parameters
st.sidebar.subheader("Generation Parameters")
top_k_questions = st.sidebar.slider("Top-K Similar Questions", min_value=1, max_value=20, value=5)
top_k_textbook = st.sidebar.slider("Top-K Textbook Chunks", min_value=1, max_value=10, value=3)
num_questions = st.sidebar.slider("Number of Questions", min_value=1, max_value=10, value=3)
difficulty = st.sidebar.selectbox("Difficulty Level", ["Easy", "Medium", "Hard"])
question_type = st.sidebar.selectbox("Question Type", ["Multiple Choice", "Short Answer", "Essay"])

# Initialize generator
if st.sidebar.button("Initialize System"):
    if questions_file or textbook_file:
        with st.spinner("Initializing question generator..."):
            config = Config()
            st.session_state.generator = QuestionGenerator(config)
            
            # Save uploaded files temporarily and initialize database
            questions_data = None
            textbook_content = None
            
            if questions_file:
                with open("temp_questions.csv", "wb") as f:
                    f.write(questions_file.getbuffer())
                questions_data = "temp_questions.csv"
            
            if textbook_file:
                with open("temp_textbook.txt", "wb") as f:
                    f.write(textbook_file.getbuffer())
                textbook_content = "temp_textbook.txt"
            
            # Initialize database
            st.session_state.generator.initialize_database(
                questions_file=questions_data,
                textbook_file=textbook_content
            )
            st.session_state.data_loaded = True
            
            # Clean up temp files
            for temp_file in ["temp_questions.csv", "temp_textbook.txt"]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        st.success("System initialized successfully!")
    else:
        st.error("Please upload at least one data source file.")

# Main interface
if st.session_state.data_loaded:
    st.header("Generate Questions")
    
    # Single question generation
    topic = st.text_input("Enter Topic", placeholder="e.g., machine learning, data structures, etc.")
    
    if st.button("Generate Questions") and topic:
        with st.spinner(f"Generating {num_questions} questions about '{topic}'..."):
            try:
                questions = []
                for i in range(num_questions):
                    question_data = st.session_state.generator.generate_new_question(
                        topic=topic,
                        difficulty=difficulty,
                        question_type=question_type,
                        top_k_questions=top_k_questions,
                        top_k_textbook=top_k_textbook
                    )
                    
                    # Evaluate question quality
                    evaluation = st.session_state.generator.llm.evaluate_question_quality(question_data)
                    question_data['evaluation'] = evaluation
                    questions.append(question_data)
                
                st.success(f"Generated {len(questions)} questions!")
                
                # Display questions
                for i, q_data in enumerate(questions, 1):
                    with st.expander(f"Question {i} (Score: {q_data.get('evaluation', {}).get('overall_score', 'N/A')}/10)"):
                        st.write("**Question:**")
                        st.write(q_data['question'])
                        
                        if 'options' in q_data and q_data['options']:
                            st.write("**Options:**")
                            for option in q_data['options']:
                                st.write(option)
                        
                        if 'correct_answer' in q_data:
                            st.write("**Correct Answer:**", q_data['correct_answer'])
                        
                        if 'explanation' in q_data:
                            st.write("**Explanation:**")
                            st.write(q_data['explanation'])
                        
                        st.write("**Context Used:**")
                        metadata = q_data.get('generation_metadata', {})
                        st.write(f"- {metadata.get('similar_questions_count', 0)} similar questions")
                        st.write(f"- {metadata.get('textbook_chunks_used', 0)} textbook chunks")
                        
                        if 'evaluation' in q_data and isinstance(q_data['evaluation'], dict):
                            st.write("**Quality Evaluation:**")
                            eval_data = q_data['evaluation']
                            if 'clarity' in eval_data:
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Clarity", f"{eval_data.get('clarity', {}).get('score', 'N/A')}/10")
                                    st.metric("Relevance", f"{eval_data.get('relevance', {}).get('score', 'N/A')}/10")
                                with col2:
                                    st.metric("Difficulty", f"{eval_data.get('difficulty', {}).get('score', 'N/A')}/10")
                                    st.metric("Educational Value", f"{eval_data.get('educational_value', {}).get('score', 'N/A')}/10")
                
                # Download option
                if st.button("Download Results as JSON"):
                    results_json = json.dumps(questions, indent=2)
                    st.download_button(
                        label="Download JSON",
                        data=results_json,
                        file_name=f"questions_{topic.replace(' ', '_')}.json",
                        mime="application/json"
                    )
                        
            except Exception as e:
                st.error(f"Error generating questions: {str(e)}")
                st.error(f"Error details: {type(e).__name__}")
    
    # Batch generation
    st.header("Batch Generation")
    topics_text = st.text_area("Enter Topics (one per line)", 
                              placeholder="machine learning\ndata structures\ndatabase design")
    
    if st.button("Generate for All Topics") and topics_text:
        topics_list = [topic.strip() for topic in topics_text.split('\n') if topic.strip()]
        
        with st.spinner(f"Generating questions for {len(topics_list)} topics..."):
            try:
                batch_results = st.session_state.generator.batch_generate_questions(
                    topics=topics_list,
                    difficulty=difficulty,
                    question_type=question_type,
                    questions_per_topic=num_questions
                )
                
                st.success(f"Generated questions for {len(topics_list)} topics!")
                
                # Display batch results
                for i, q_data in enumerate(batch_results, 1):
                    topic = q_data.get('topic', f'Question {i}')
                    st.subheader(f"📚 {topic}")
                    st.write(f"**Question:** {q_data['question']}")
                    if 'options' in q_data and q_data['options']:
                        for option in q_data['options']:
                            st.write(option)
                    st.write("---")
                
                # Download batch results
                batch_json = json.dumps(batch_results, indent=2)
                st.download_button(
                    label="Download All Results",
                    data=batch_json,
                    file_name="batch_generated_questions.json",
                    mime="application/json"
                )
                
            except Exception as e:
                st.error(f"Error in batch generation: {str(e)}")

else:
    st.info("👆 Please upload data files and initialize the system using the sidebar.")
    
    # Show example data formats
    st.header("📋 Data Format Examples")
    
    st.subheader("Questions CSV Format")
    st.code("""question,answer,topic,difficulty
"What is machine learning?","A subset of AI that enables computers to learn without explicit programming","machine learning","easy"
"Explain gradient descent","An optimization algorithm used to minimize cost functions","machine learning","medium"
""")
    
    st.subheader("Questions JSON Format")
    st.code("""{
  "questions": [
    {
      "question": "What is a binary search tree?",
      "answer": "A tree data structure where left child < parent < right child",
      "topic": "data structures",
      "difficulty": "medium"
    }
  ]
}""")
    
    st.subheader("Textbook Content Format")
    st.code("""Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data...

Data structures are ways of organizing and storing data in a computer so that it can be accessed and modified efficiently...

Chapter 1: Introduction to Algorithms
An algorithm is a step-by-step procedure for solving a problem or completing a task...""")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, and OpenAI")