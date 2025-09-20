import openai
from typing import List, Dict, Any
import json

class LLMIntegration:
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        openai.api_key = api_key
        self.model = model
    
    def generate_question_prompt(self, 
                               topic: str,
                               similar_questions: List[Dict[str, Any]],
                               textbook_content: List[Dict[str, Any]],
                               difficulty: str = "Medium",
                               question_type: str = "Multiple Choice") -> str:
        """Create a comprehensive prompt for question generation"""
        
        prompt = f"""You are an expert question generator. Your task is to create a new, unique {question_type.lower()} question about {topic} at {difficulty.lower()} difficulty level.

CONTEXT FROM SIMILAR QUESTIONS:
"""
        
        for i, q in enumerate(similar_questions[:3], 1):
            prompt += f"\n{i}. Question: {q['question']}\n   Answer: {q['answer']}\n   Topic: {q['topic']}\n"
        
        prompt += "\nRELEVANT TEXTBOOK CONTENT:\n"
        for i, content in enumerate(textbook_content, 1):
            prompt += f"\n{i}. {content['content'][:300]}...\n"
        
        prompt += f"""
REQUIREMENTS:
1. Create a NEW question that is similar in style but different in content from the examples above
2. The question should be at {difficulty.lower()} difficulty level
3. Base the question on the textbook content provided
4. Make it a {question_type.lower()} question
5. Ensure the question tests understanding, not just memorization
6. Provide a clear, accurate answer

OUTPUT FORMAT:
{{
    "question": "Your generated question here",
    "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
    "correct_answer": "A",
    "explanation": "Detailed explanation of why this is the correct answer",
    "topic": "{topic}",
    "difficulty": "{difficulty}",
    "question_type": "{question_type}"
}}

Generate the question now:"""
        
        return prompt
    
    def generate_question(self, 
                         topic: str,
                         similar_questions: List[Dict[str, Any]],
                         textbook_content: List[Dict[str, Any]],
                         difficulty: str = "Medium",
                         question_type: str = "Multiple Choice",
                         max_tokens: int = 500,
                         temperature: float = 0.7) -> Dict[str, Any]:
        """Generate a new question using the LLM"""
        
        prompt = self.generate_question_prompt(
            topic, similar_questions, textbook_content, difficulty, question_type
        )
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert educational content creator specializing in question generation."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            generated_content = response.choices[0].message.content.strip()
            
            # Try to parse JSON response
            try:
                question_data = json.loads(generated_content)
                return question_data
            except json.JSONDecodeError:
                # If JSON parsing fails, return raw content
                return {
                    "question": generated_content,
                    "raw_response": True,
                    "topic": topic,
                    "difficulty": difficulty,
                    "question_type": question_type
                }
                
        except Exception as e:
            return {
                "error": f"Failed to generate question: {str(e)}",
                "topic": topic,
                "difficulty": difficulty,
                "question_type": question_type
            }
    
    def evaluate_question_quality(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate the quality of a generated question"""
        
        evaluation_prompt = f"""Evaluate the quality of this generated question on a scale of 1-10:

Question: {question_data.get('question', '')}
Options: {question_data.get('options', [])}
Correct Answer: {question_data.get('correct_answer', '')}
Explanation: {question_data.get('explanation', '')}

Rate the question on:
1. Clarity (1-10)
2. Difficulty appropriateness (1-10)
3. Educational value (1-10)
4. Answer accuracy (1-10)
5. Overall quality (1-10)

Provide brief feedback for improvement.

Format your response as JSON:
{{
    "clarity": score,
    "difficulty": score,
    "educational_value": score,
    "accuracy": score,
    "overall": score,
    "feedback": "Brief feedback here"
}}
"""
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an educational assessment expert."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            evaluation_content = response.choices[0].message.content.strip()
            return json.loads(evaluation_content)
            
        except Exception as e:
            return {"error": f"Failed to evaluate question: {str(e)}"}