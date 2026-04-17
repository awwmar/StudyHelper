import json
import logging
import re
from services.vector import vector_service
from services.selector import selector_agent  # Using the existing cross-encoder
from services.llm import llm_service
from services.db import DBService
from config import config
from services.knowledge_graph import knowledge_graph_service
import concurrent.futures

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
        
        # Topic Boost
        if topics:
            topic_chunks = []
            for t in topics:
                c_list = DBService.get_chunks_by_topic(t)
                topic_chunks.extend(c_list[:3])
            chunks.extend(topic_chunks)
            
        # Knowledge Graph Boost
        if config.ENABLE_KNOWLEDGE_GRAPH:
            concepts = knowledge_graph_service.extract_query_concepts(query)
            if concepts:
                kg_chunks = DBService.get_kg_related_chunks(concepts)
                chunks.extend(kg_chunks)
            
        # Deduplicate
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
        # We wrap the existing CrossEncoder selector
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
        
        Return ONLY a valid JSON array of strings. Example: ["Question 1?", "Question 2?"]
        Do not add numbering or markdown blocks.
        
        Context:
        {context[:6000]}
        """
        return llm_service.invoke_custom(prompt)

class EvaluatorAgent:
    """Agent 4: Checks Exact Match, Faithfulness, and Hallucination metrics."""
    def evaluate(self, context: str, drafted_content: str) -> dict:
        logger.info("EvaluatorAgent: Checking output for Faithfulness & Hallucinations...")
        prompt = f"""
        You are an expert Evaluator Agent.
        
        Original Context:
        {context[:6000]}
        
        Drafted Output:
        {drafted_content}
        
        Evaluate the drafted content:
        1. JSON Format: Is it a valid JSON array of strings?
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
        
        # 🔁 Cache Mechanism
        self.cache = {}

    def _get_cache_key(self, topics, difficulty, num, q_type):
        return f"{'-'.join(sorted(topics))}_{difficulty}_{num}_{q_type}"

    def generate_quiz(self, topics: list, difficulty: str, num_questions: int, q_type: str, type_instruction: str) -> list:
        cache_key = self._get_cache_key(topics, difficulty, num_questions, q_type)
        
        # Cache Check
        if cache_key in self.cache:
            logger.info("System 2 Cache Hit! Reusing previous quiz generation.")
            return self.cache[cache_key]

        query = f"Generate {difficulty} questions about {', '.join(topics)} of type {q_type}"
        
        # [Retriever Agent]
        chunks = self.retriever_agent.retrieve(query, topics)
        
        # [Selector Agent]
        selected_chunks = self.selector_agent.filter_chunks(query, chunks)
        context = "\n\n".join([c['text_content'] for c in selected_chunks])
        
        # [Response -> Evaluator Loop]
        feedback = ""
        final_draft = ""
        max_retries = 3
        
        for attempt in range(max_retries):
            # [Response Agent]
            drafted = self.response_agent.draft_output(context, topics, difficulty, num_questions, type_instruction, feedback)
            final_draft = drafted
            
            # [Evaluator Agent]
            eval_result = self.evaluator_agent.evaluate(context, drafted)
            if eval_result.get("pass", False):
                break
            else:
                feedback = eval_result.get("feedback", "")
                
        # Parse final result
        questions_list = []
        try:
            cleaned = re.search(r'\[.*\]', final_draft, re.DOTALL)
            questions_list = json.loads(cleaned.group(0)) if cleaned else json.loads(final_draft)
            questions_list = [str(q).strip() for q in questions_list if q]
        except:
            questions_list = [line.strip() for line in final_draft.split('\n') if '?' in line]
            
        if not questions_list:
            questions_list = ["Error generating questions."]
            
        questions_list = questions_list[:num_questions]
        
        # Update Cache
        self.cache[cache_key] = questions_list
        logger.info("System 2 Complete: Output Cached.")
        
        return questions_list

system2 = System2Orchestrator()
