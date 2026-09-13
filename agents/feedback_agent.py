import logging
from services.llm import llm_service

logger = logging.getLogger(__name__)

class FeedbackAgent:
    """A Socratic tutor that leads students to answers without giving them away."""

    def generate_socratic_hint(self, context: str, question: str, student_attempt: str) -> str:
        logger.info("FeedbackAgent: Drafting Socratic hint...")
        prompt = f"""
        You are an expert, encouraging Socratic tutor for an engineering student.
        The student is trying to answer a question but is struggling.

        Context Text:
        {context[:3000]}

        Question: {question}
        Student's current thought process or answer: {student_attempt}

        DO NOT provide the direct answer. Instead, provide a guiding hint or a counter-question
        to help the student figure out the correct logic path. Be brief and supportive.
        """
        return llm_service.invoke_custom(prompt)

feedback_agent = FeedbackAgent()
