import json
import logging
import re
from services.vector import vector_service
from services.selector import selector_agent
from services.llm import llm_service
from services.db import DBService
from config import config
from services.knowledge_graph import knowledge_graph_service
import concurrent.futures

# Import new agents
from agents.caching_agent import caching_agent
from agents.profiling_agent import profiling_agent
from agents.distractor_agent import distractor_agent
from agents.prompt_agent import prompt_optimization_agent

logger = logging.getLogger(__name__)

class RetrieverAgent:
    """Agent 1: Searches vector DB and retrieves top relevant chunks."""
    def retrieve(self, query: str, topics: list = None, top_k: int = 30) -> list:
        logger.info("RetrieverAgent: Searching Vector DB for context...")
        chunks = []
        if config.ENABLE_HYBRID_SEARCH:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                faiss_future = executor.submit(vector_service.search, query, top_k)
                bm25_future = executor.submit(vector_service.bm25_search, query, top_k)
                
                faiss_chunks = faiss_future.result()
                bm25_chunks = bm25_future.result()
                
                # Reciprocal Rank Fusion (RRF)
                k_rrf = 60
                rrf_scores = {}
                chunk_map = {}
                
                for rank, chunk in enumerate(faiss_chunks):
                    cid = chunk['id']
                    chunk_map[cid] = chunk
                    if cid not in rrf_scores:
                        rrf_scores[cid] = 0
                    rrf_scores[cid] += 1 / (k_rrf + rank + 1)
                    
                for rank, chunk in enumerate(bm25_chunks):
                    cid = chunk['id']
                    chunk_map[cid] = chunk
                    if cid not in rrf_scores:
                        rrf_scores[cid] = 0
                    rrf_scores[cid] += 1 / (k_rrf + rank + 1)
                    
                sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
                chunks = [chunk_map[cid] for cid in sorted_cids[:top_k]]
        else:
            chunks = vector_service.search(query, k=top_k)
        
        if topics:
            topic_chunks = []
            for t in topics:
                c_list = DBService.get_chunks_by_topic(t)
                topic_chunks.extend(c_list[:3])
            chunks.extend(topic_chunks)
            
        if config.ENABLE_KNOWLEDGE_GRAPH:
            concepts = knowledge_graph_service.extract_query_concepts(query)
            if concepts:
                kg_chunks = DBService.get_kg_related_chunks(concepts)
                chunks.extend(kg_chunks)
            
        seen = set()
        unique_chunks = []
        for c in chunks:
            text = c['text_content']
            if text not in seen:
                seen.add(text)
                unique_chunks.append(c)
                
        return unique_chunks

class SelectorAgentProxy:
    """Agent 2: Filters irrelevant chunks, keeping top 50% useful chunks."""
    def __init__(self):
        self.selector = selector_agent

    def filter_chunks(self, query: str, chunks: list, top_ratio: float = 0.5) -> list:
        initial_count = len(chunks)
        if config.ENABLE_CHUNK_LENGTH_FILTER:
            chunks = [c for c in chunks if len(c['text_content']) >= 300]
            dropped = initial_count - len(chunks)
            if dropped > 0:
                logger.info(f"SelectorAgentProxy: Dropped {dropped} chunks because length < 300")
                
        logger.info(f"SelectorAgent: Filtering {len(chunks)} down to {int(len(chunks)*top_ratio)} chunks...")
        return self.selector.select(query, chunks, top_ratio=top_ratio, min_chunks=5)


class ResponseAgent:
    """Agent 3: Calls LLM with cached context to generate answers/MCQ."""
    def draft_output(self, context: str, topics: list, difficulty: str, num_questions: int, type_instruction: str, feedback: str = "") -> str:
        logger.info("ResponseAgent: Drafting MCQs from context...")
        feedback_prompt = f"\nCRITICAL FEEDBACK FROM EVALUATOR AGENT:\n{feedback}\n" if feedback else ""
        
        prompt = f"""
        Create a {difficulty} level test with {num_questions} questions based exclusively on the provided context.
        Focus on topics: {topics}.
        {type_instruction}
        {feedback_prompt}
        
        Return ONLY a valid JSON array of objects. Format:
        [
            {{"question": "What is ...?", "correct": "The exact answer is ..."}},
            {{"question": "How does ...?", "correct": "By doing ..."}}
        ]
        Do not add distractors yet. Do not add markdown code blocks like ```json.
        
        Context:
        {context[:8000]}
        """
        return llm_service.invoke_custom(prompt)


class EvaluatorAgent:
    """Agent 4: Checks Exact Match, Faithfulness, and Hallucination metrics."""
    def evaluate(self, context: str, drafted_content: str) -> dict:
        logger.info("EvaluatorAgent: Checking output for Faithfulness & Hallucinations...")
        prompt = f"""
        You are an expert Evaluator Agent.
        
        Original Context:
        {context[:8000]}
        
        Drafted Output:
        {drafted_content}
        
        Evaluate the drafted content:
        1. JSON Format: Is it a valid JSON array of objects with keys 'question' and 'correct'?
        2. Faithfulness/No Hallucinations: Can every single question be answered fully using only the 'Original Context'?
        
        Return your evaluation exactly as a JSON object:
        {{
            "pass": true / false,
            "feedback": "Detail exactly which questions are hallucinatory or have formatting errors."
        }}
        """
        response = llm_service.invoke_custom(prompt)
        try:
            cleaned = re.search(r'\{.*\}', response, re.DOTALL)
            return json.loads(cleaned.group(0)) if cleaned else json.loads(response)
        except Exception as e:
            return {"pass": False, "feedback": "JSON output malformed."}


class System2Orchestrator:
    """Orchestrates System 2 Query Pipeline with caching mechanism."""
    def __init__(self):
        self.retriever_agent = RetrieverAgent()
        self.selector_agent = SelectorAgentProxy()
        self.response_agent = ResponseAgent()
        self.evaluator_agent = EvaluatorAgent()
        # Profiling, Caching, PromptOptimization, and Distractor imported directly.
        
    def generate_quiz(self, topics: list, difficulty: str, num_questions: int, q_type: str, type_instruction: str, user_id: str = "default") -> list:
        # Cache Check Agent
        cached = caching_agent.get_cache(topics, difficulty, num_questions, q_type)
        if cached:
            return cached

        # Profile Agent adjusts instructions
        profile_instruction = profiling_agent.get_learner_context(user_id, topics)
        enhanced_instruction = type_instruction + "\n" + profile_instruction

        query = f"Generate {difficulty} questions about {', '.join(topics)} of type {q_type}"
        chunks = self.retriever_agent.retrieve(query, topics)
        selected_chunks = self.selector_agent.filter_chunks(query, chunks)
        
        # Prompt Optimization Agent mitigates Lost-in-the-Middle
        context = prompt_optimization_agent.format_context(selected_chunks)

        feedback = ""
        final_draft = ""
        max_retries = 3
        questions_list = []
        
        for attempt in range(max_retries):
            drafted = self.response_agent.draft_output(context, topics, difficulty, num_questions, enhanced_instruction, feedback)
            final_draft = drafted
            eval_result = self.evaluator_agent.evaluate(context, drafted)
            if eval_result.get("pass", False):
                break
            else:
                feedback = eval_result.get("feedback", "")
                
        # Parse exact answers
        try:
            cleaned = re.search(r'\[.*\]', final_draft, re.DOTALL)
            questions_list = json.loads(cleaned.group(0)) if cleaned else json.loads(final_draft)
        except:
            questions_list = []
            
        if not questions_list:
            questions_list = [{"question": "Error generating questions.", "correct": "System failure."}]
            
        questions_list = questions_list[:num_questions]
        
        # Distractor Agent fetches wrong answers
        if q_type.lower() == "mcq" or "mcq" in type_instruction.lower():
            final_quiz = distractor_agent.generate_distractors(context, questions_list)
        else:
            final_quiz = questions_list

        caching_agent.store_cache(topics, difficulty, num_questions, q_type, final_quiz)
        logger.info("System 2 Complete: Output Cached.")
        return final_quiz
        
    def cache_check_only(self, query: str) -> bool:
        """Simple helper for demo purposes to check if a generic query exists in cache."""
        # For the demo, we'll just check if there's *any* cache entry for basic topics
        # Real cache check is inside generate_quiz
        return False # Default to miss for demo trace unless we want to be specific

system2 = System2Orchestrator()
