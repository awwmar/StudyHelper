import logging
from services.db import DBService

logger = logging.getLogger(__name__)

class ProfilingAgent:
    """Agent that assesses user history and modifies the query generation strategy accordingly."""

    def get_learner_context(self, user_id: str, topics: list) -> str:
        """Returns specialized LLM instructions based on user mastery levels."""
        if not user_id or not topics:
            return ""

        instruction = "\n### LEARNER PROFILE INSTRUCTIONS ###\n"
        for topic in topics:
            mastery = DBService.get_learner_profile(user_id, topic)
            if mastery < 0.4:
                instruction += f"- The student struggles significantly with '{topic}'. Use extremely simple, foundational language and exhaustive explanations.\n"
            elif mastery > 0.8:
                instruction += f"- The student has high mastery of '{topic}'. Use advanced, terse phrasing and focus on edge-case challenging scenarios.\n"
            else:
                instruction += f"- The student is relatively average proficient in '{topic}'. Standard phrasing is appropriate.\n"
        return instruction + "####################################\n"

profiling_agent = ProfilingAgent()
