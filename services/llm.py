import os
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import cachetools

load_dotenv()
logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in env")
            
        self.llm = ChatGroq(
            temperature=0.3, # Lower temperature for reasoning/extraction
            model_name="llama-3.1-8b-instant",
            groq_api_key=api_key,
            max_retries=1
        )
        # Fallback model
        self.fallback_llm = ChatGroq(
            temperature=0.3,
            model_name="llama-3.1-8b-instant",
            groq_api_key=api_key
        )

    def extract_metadata(self, text_chunk: str):
        """
        Extracts a summary (1-2 sentences) and 3-5 topics from the text.
        Returns a dict: {"summary":Str, "topics":List[Str]}
        """
        prompt = f"""
You are an expert content analyzer.
Analyze the following text chunk and extract:
1. A concise summary (1-2 sentences).
2. A list of 3-5 specific topics or keywords covered in this chunk.

Return ONLY valid JSON in the following format:
{{
  "summary": "...",
  "topics": ["topic1", "topic2", "topic3"]
}}

Text Chunk:
{text_chunk[:3000]} # Limit context window if needed
"""
        try:
            response = self.llm.invoke(prompt).content
            # Basic cleanup if model adds ```json ... ```
            clean_response = response.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_response)
            return data
        except Exception as e:
            logger.warning(f"LLM Extraction failed (Primary): {e}. Trying fallback.")
            try:
                response = self.fallback_llm.invoke(prompt).content
                clean_response = response.replace("```json", "").replace("```", "").strip()
                data = json.loads(clean_response)
                return data
            except Exception as e2:
                logger.error(f"LLM Extraction failed (Fallback): {e2}")
                return {"summary": "Summary not available.", "topics": []}

    @cachetools.cached(cache=cachetools.TTLCache(maxsize=100, ttl=3600))
    def generate_response(self, query: str, context: str):
        prompt = f"""
Answer the user query based ONLY on the context provided below.
If the answer is not in the context, say "I don't know based on the provided documents."

Context:
{context}

User Query: {query}
"""
        
        try:
            return self.llm.invoke(prompt).content
        except Exception as e:
            logger.warning(f"LLM Generation failed (Primary): {e}. Trying fallback.")
            return self.fallback_llm.invoke(prompt).content

    def evaluate_answer(self, question: str, student_answer: str, reference_context: str = None, teacher_review: str = None) -> str:
        """
        Evaluates a student answer using the Master StudyHub Grading Prompt.
        Returns a string with "Score: X | Feedback: ..."
        """
        
        # --- BLANK ANSWER DETECTION (in Python, not LLM) ---
        blank_indicators = ["", "no answer provided", "i don't know", "n/a", "none", "idk", "na", "no answer"]
        cleaned_answer = (student_answer or "").strip().lower()
        if cleaned_answer in blank_indicators or len(cleaned_answer) == 0:
            return json.dumps({
                "correctness": "incorrect",
                "score": 0,
                "final_answer_correctness": 0,
                "logical_validity": 0,
                "conceptual_accuracy": 0,
                "completeness": 0,
                "issues": ["No answer provided."],
                "feedback": "No answer was provided."
            })

        # Build reference section
        reference_section = ""
        if reference_context:
            reference_section = f"""
Reference Answer / Textbook Context (Ground Truth):
{reference_context}
"""

        # Build teacher review section
        teacher_section = ""
        if teacher_review:
            teacher_section = f"""
[Teacher's Manual Review/Feedback]:
{teacher_review}

CRITICAL INSTRUCTION:
- The teacher has manually reviewed this answer.
- You MUST prioritize the teacher's judgment above all else.
- If the teacher says it's correct, give a high score.
- If the teacher says it's incorrect, give a low score.
- Incorporate the teacher's feedback into your response.
"""

        prompt = f"""You are an expert evaluator for technical subjects, especially:

• Mathematics
• Computer Science
• Algorithms / Proofs / Problem Solving

Your job is to grade answers with high emphasis on correctness, logic, and reasoning validity.

Inputs

Question:
{question}

Teacher Answer Key / Reference Context:
{reference_context if reference_context else 'None provided.'}

Student Answer:
{student_answer}

{teacher_section}

🎯 Evaluation Priorities (ORDERED)

1️⃣ Final Answer Correctness (If Applicable)
If the question expects a numerical value, formula, output, complexity, or code result, check if the final result is correct. Incorrect final answer -> significant penalty.

2️⃣ Logical / Mathematical Validity (CRITICAL)
Evaluate valid reasoning steps, sound mathematical logic, correct algorithmic thinking, and no invalid inferences.
- Fluent but logically wrong -> grade low
- Correct-looking conclusion with invalid steps -> penalize
- Different but valid method -> accept

3️⃣ Conceptual Accuracy
Check for correct definitions, correct theorem usage, and correct CS principles.

4️⃣ Completeness
Did the student include required steps, necessary justification, and edge cases (if relevant)? Missing reasoning -> partial credit.

5️⃣ Alternative Correct Approaches
Accept equivalent formulas, different valid proofs, different correct algorithms, and different correct complexity derivations.

6️⃣ Error & Misconception Detection
Identify mathematical errors, wrong formulas, invalid assumptions, and incorrect CS concepts. Major conceptual error -> strong penalty.

7️⃣ Use of RAG Context + Domain Knowledge
Use retrieved material for grounding BUT allow correct external knowledge and flag contradictions with known theory.

⚠️ Strict Technical Rules
❌ Do NOT reward verbosity without correctness
❌ Do NOT accept hand-wavy reasoning
❌ Do NOT ignore incorrect formulas
✅ Precision matters
✅ Logic matters
✅ Correctness matters

📊 Scoring Scheme (0 – 10)
9–10 → Correct answer + valid reasoning
7–8 → Correct core logic, minor mistakes
4–6 → Partial correctness / gaps
1–3 → Mostly incorrect / flawed logic
0 → Completely wrong / nonsense

✅ Output Format (STRICT JSON ONLY)
Return ONLY valid JSON.

{{
  "correctness": "correct / partial / incorrect",
  "score": 0,
  "final_answer_correctness": 0,
  "logical_validity": 0,
  "conceptual_accuracy": 0,
  "completeness": 0,
  "issues": [],
  "feedback": ""
}}

🧾 Field Meaning
• final_answer_correctness → Accuracy of result (0–10)
• logical_validity → Validity of steps/proofs/logic (0–10)
• conceptual_accuracy → Correct theory usage (0–10)
• completeness → Coverage of required steps (0–10)

🧠 Evaluator Mindset
Act like a strict but fair professor: Reward correctness and valid reasoning, accept valid alternatives, penalize logical errors, and penalize incorrect math/CS concepts.
"""
        try:
            response = self.llm.invoke(prompt).content
            return response.replace("```json", "").replace("```", "").strip()
        except Exception as e:
            logger.warning(f"LLM Evaluation failed (Primary): {e}. Trying fallback.")
            response = self.fallback_llm.invoke(prompt).content
            return response.replace("```json", "").replace("```", "").strip()

    def invoke_custom(self, prompt: str) -> str:
        """Generic method for custom prompts (used by eval_metrics, etc.)."""
        try:
            return self.llm.invoke(prompt).content
        except Exception as e:
            logger.warning(f"LLM invoke failed (Primary): {e}. Trying fallback.")
            return self.fallback_llm.invoke(prompt).content

    def evaluate_batch(self, items: list, max_workers: int = 5) -> list:
        """
        Evaluates multiple answers in parallel using ThreadPoolExecutor.
        
        Args:
            items: List of dicts, each with keys:
                - question (str)
                - student_answer (str)
                - reference_context (str, optional)
                - teacher_review (str, optional)
            max_workers: Max concurrent LLM calls (be careful with rate limits)
        
        Returns:
            List of evaluation result strings, in the same order as input
        """
        results = [None] * len(items)

        def _evaluate_single(index, item):
            result = self.evaluate_answer(
                question=item['question'],
                student_answer=item['student_answer'],
                reference_context=item.get('reference_context'),
                teacher_review=item.get('teacher_review')
            )
            return index, result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_evaluate_single, i, item): i 
                for i, item in enumerate(items)
            }
            for future in as_completed(futures):
                try:
                    index, result = future.result()
                    results[index] = result
                except Exception as e:
                    idx = futures[future]
                    logger.error(f"Parallel evaluation failed for item {idx}: {e}")
                    results[idx] = json.dumps({
                        "correctness": "error",
                        "score": 0,
                        "feedback": f"Evaluation failed: {str(e)}"
                    })

        logger.info(f"Batch evaluation complete: {len(items)} items evaluated")
        return results

llm_service = LLMService()

