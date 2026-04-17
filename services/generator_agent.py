import json
import logging
import re
from services.llm import llm_service

logger = logging.getLogger(__name__)

class GeneratorAgent:
    """
    Agentic workflow for generating test questions.
    Implements a Draft -> Critique -> Refine loop.
    Ensure strict faithfulness to context and strictly valid JSON output.
    """
    
    def __init__(self, max_retries=3):
        self.max_retries = max_retries

    def draft_questions(self, context: str, topics: list, difficulty: str, num_questions: int, type_instruction: str, previous_feedback: str = "") -> str:
        """Helper to generate draft questions."""
        
        feedback_prompt = ""
        if previous_feedback:
            feedback_prompt = f"\nCRITICAL FEEDBACK FROM CRITIQUE AGENT (FIX THESE ISSUES):\n{previous_feedback}\n"
            
        prompt = f"""
        Create a {difficulty} level test with {num_questions} questions based exclusively on the provided context.
        Focus on topics: {topics}.
        {type_instruction}
        {feedback_prompt}
        
        Return ONLY a valid JSON array of strings, where each string is a complete question.
        Example: ["Question 1 text?", "Question 2 text?"]
        Do not add numbering, markdown blocks like ```json, or any conversational text.
        
        Context:
        {context[:6000]}
        """
        return llm_service.invoke_custom(prompt)

    def critique_questions(self, context: str, drafted_content: str) -> dict:
        """
        Critique agent that reviews the drafted questions for:
        1. JSON Validity
        2. Faithfulness to context (No Hallucinations)
        """
        
        prompt = f"""
        You are an expert Review Agent. Your job is to review the drafted test questions for a student quiz.
        
        Original Context:
        {context[:6000]}
        
        Drafted Questions:
        {drafted_content}
        
        Evaluate the drafted questions on these two criteria:
        1. JSON Format: Is it a pure JSON array of strings?
        2. Faithfulness: Can every single question be answered fully using only the 'Original Context' provided above?
        
        Return your evaluation exactly as a JSON object:
        {{
            "pass": true / false,
            "feedback": "If pass is false, explain exactly which questions are hallucinatory or what formatting error exists."
        }}
        Do not include any other text.
        """
        
        response = llm_service.invoke_custom(prompt)
        
        # Parse the critique
        try:
            cleaned = re.search(r'\{.*\}', response, re.DOTALL)
            if cleaned:
                critique = json.loads(cleaned.group(0))
            else:
                critique = json.loads(response)
            return critique
        except Exception as e:
            logger.warning(f"Critique parsing failed: {e}. Assuming failure.")
            return {"pass": False, "feedback": f"Critique agent output could not be parsed as JSON. Output was: {response[:100]}"}

    def generate_with_critique(self, context: str, topics: list, difficulty: str, num_questions: int, type_instruction: str) -> list:
        """
        Executes the agentic loop: Draft -> Critique -> Refine (up to max_retries).
        """
        
        feedback = ""
        final_draft = ""
        
        for attempt in range(1, self.max_retries + 1):
            logger.info(f"GeneratorAgent: Drafting questions (Attempt {attempt}/{self.max_retries})")
            
            # 1. Draft
            drafted_content = self.draft_questions(context, topics, difficulty, num_questions, type_instruction, feedback)
            final_draft = drafted_content # Save latest attempt
            
            # 2. Critique
            logger.info(f"GeneratorAgent: Critiquing questions...")
            critique = self.critique_questions(context, drafted_content)
            
            if critique.get("pass", False):
                logger.info("GeneratorAgent: Critique passed!")
                break
            else:
                feedback = critique.get("feedback", "Failed criteria.")
                logger.warning(f"GeneratorAgent: Critique failed. Feedback: {feedback}")
                if attempt == self.max_retries:
                    logger.warning("GeneratorAgent: Max retries reached. Returning last draft.")
        
        # Attempt to parse final_draft as JSON array
        questions_list = []
        try:
            cleaned = re.search(r'\[.*\]', final_draft, re.DOTALL)
            if cleaned:
                questions_list = json.loads(cleaned.group(0))
            else:
                questions_list = json.loads(final_draft)
            
            if not isinstance(questions_list, list):
                raise ValueError("Parsed JSON is not a list.")
                
            questions_list = [str(q).strip() for q in questions_list if q]
            
        except Exception as e:
            logger.error(f"GeneratorAgent: Final fallback parsing due to error: {e}")
            lines = final_draft.split('\n')
            for line in lines:
                line = line.strip()
                line = re.sub(r'^[\d\-\.\)]+\s*', '', line)
                if line and '?' in line:
                    questions_list.append(line)
        
        if not questions_list:
            questions_list = ["Error generating questions. Please try again or check context availability."]
            
        # Ensure we return exact amount requested
        return questions_list[:num_questions]

generator_agent = GeneratorAgent()
