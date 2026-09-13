import logging
import json
import re
from services.llm import llm_service

logger = logging.getLogger(__name__)

class DistractorAgent:
    """Generates highly plausible but systematically incorrect distractor choices."""

    def generate_distractors(self, context: str, exact_questions: list) -> list:
        """
        Takes a list of correct questions/answers and generates 3 distractors for each.
        Input exact_questions format: [{"question": "...", "correct": "..."}, ...]
        Returns: [{"question": "...", "correct": "...", "distractors": ["...", "...", "..."]}]
        """
        logger.info("DistractorAgent: Generating plausible distractors...")

        prompt = f"""
        You are a Professor producing tough multiple choice tests.
        I am giving you a list of questions and their correct answers.
        For each question, generate EXACTLY 3 plausible but fundamentally incorrect choices (distractors) based on common student misconceptions.

        Context for reference:
        {context[:3000]}

        Input List:
        {json.dumps(exact_questions)}

        Output MUST be a JSON array of objects mimicking the input but with a new 'distractors' key:
        [
            {{
                "question": "...",
                "correct": "...",
                "distractors": ["wrong1", "wrong2", "wrong3"]
            }}
        ]
        """

        response = llm_service.invoke_custom(prompt)
        try:
            cleaned = re.search(r'\[.*\]', response, re.DOTALL)
            payload = json.loads(cleaned.group(0)) if cleaned else json.loads(response)
            return payload
        except Exception as e:
            logger.error(f"DistractorAgent parsing error: {e}")
            # Fallback
            for q in exact_questions:
                q["distractors"] = ["Option A", "Option B", "Option C"]
            return exact_questions

distractor_agent = DistractorAgent()
