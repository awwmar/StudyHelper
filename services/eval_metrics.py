import json
import re
import logging
import time
import concurrent.futures
from difflib import SequenceMatcher
from services.llm import llm_service
from config import config

logger = logging.getLogger(__name__)

# Attempt to import bert_score and rouge_score
try:
    from bert_score import score as bert_score_calc
except ImportError:
    bert_score_calc = None

try:
    from rouge_score import rouge_scorer
except ImportError:
    rouge_scorer = None


def normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r'\s+', ' ', text)
    # Remove common punctuation for comparison
    text = re.sub(r'[^\w\s]', '', text)
    return text


def exact_match(generated: str, reference: str) -> float:
    """
    Computes exact match score between generated and reference answers.

    Returns:
        1.0 if normalized texts match exactly, else a partial similarity score (0-1)
    """
    norm_gen = normalize_text(generated)
    norm_ref = normalize_text(reference)

    if not norm_gen or not norm_ref:
        return 0.0

    if norm_gen == norm_ref:
        return 1.0

    # Partial match using SequenceMatcher for near-matches
    return SequenceMatcher(None, norm_gen, norm_ref).ratio()


def faithfulness(answer: str, context: str) -> float:
    """
    Evaluates whether the answer is grounded in the provided context.
    Uses LLM to judge if answer claims are supported by context.

    Returns:
        Score from 0.0 (hallucinated) to 1.0 (fully grounded)
    """
    if not answer or not context:
        return 0.0

    prompt = f"""You are a faithfulness evaluator. Your job is to check if an answer is fully supported by the given context.

Context:
{context[:4000]}

Answer to evaluate:
{answer[:2000]}

Evaluate: What fraction of claims in the answer are directly supported by the context?

Return ONLY valid JSON:
{{
  "supported_claims": <number of claims supported by context>,
  "total_claims": <total number of factual claims in the answer>,
  "score": <float from 0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}}"""

    try:
        response = llm_service.invoke_custom(prompt)
        clean = response.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            score = float(data.get("score", 0.0))
            return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"Faithfulness evaluation failed: {e}")

    return 0.0


def relevance(answer: str, question: str) -> float:
    """
    Evaluates whether the answer is relevant to the question asked.
    Uses LLM to judge answer relevance.

    Returns:
        Score from 0.0 (irrelevant) to 1.0 (highly relevant)
    """
    if not answer or not question:
        return 0.0

    prompt = f"""You are a relevance evaluator. Judge whether the answer addresses the question.

Question:
{question[:1000]}

Answer:
{answer[:2000]}

Rate how relevant the answer is to the question on a scale of 0.0 to 1.0:
- 1.0 = directly and fully answers the question
- 0.5 = partially relevant, addresses some aspects
- 0.0 = completely irrelevant or off-topic

Return ONLY valid JSON:
{{
  "score": <float from 0.0 to 1.0>,
  "reasoning": "<brief explanation>"
}}"""

    try:
        response = llm_service.invoke_custom(prompt)
        clean = response.replace("```json", "").replace("```", "").strip()
        json_match = re.search(r'\{.*\}', clean, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            score = float(data.get("score", 0.0))
            return max(0.0, min(1.0, score))
    except Exception as e:
        logger.warning(f"Relevance evaluation failed: {e}")

    return 0.0


def get_bert_score(generated_text: str, reference_text: str) -> float:
    if not bert_score_calc or not generated_text or not reference_text:
         return 0.0
    try:
        # Note: roberta-large can be heavy for CPU running. Setting it to distilbert-base-uncased.
        P, R, F1 = bert_score_calc([generated_text], [reference_text], lang="en", model_type="roberta-large")
        return round(float(F1[0]), 4)
    except Exception as e:
        logger.warning(f"BERTScore failed: {e}")
        return 0.0

def get_rouge_scores(generated_text: str, reference_text: str) -> dict:
    if not rouge_scorer or not generated_text or not reference_text:
         return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    try:
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        scores = scorer.score(reference_text, generated_text)
        return {
            "rouge1": round(scores['rouge1'].fmeasure, 4),
            "rouge2": round(scores['rouge2'].fmeasure, 4),
            "rougeL": round(scores['rougeL'].fmeasure, 4)
        }
    except Exception as e:
        logger.warning(f"ROUGE failed: {e}")
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}


def evaluate_all(answer: str, question: str, reference: str = None, context: str = None) -> dict:
    """
    Runs all available metrics and returns a combined result.
    Parallel execution available via config feature flag.
    """
    start_time = time.perf_counter()
    results = {}
    
    if config.ENABLE_PARALLEL_METRICS:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_metric = {}
            if reference:
                future_to_metric[executor.submit(exact_match, answer, reference)] = "exact_match"
                future_to_metric[executor.submit(get_bert_score, answer, reference)] = "bert_score"
                future_to_metric[executor.submit(get_rouge_scores, answer, reference)] = "rouge_scores"
                
            if context:
                future_to_metric[executor.submit(faithfulness, answer, context)] = "faithfulness"
                
            future_to_metric[executor.submit(relevance, answer, question)] = "relevance"
            
            for future in concurrent.futures.as_completed(future_to_metric):
                metric_name = future_to_metric[future]
                try:
                    res = future.result()
                    if metric_name == "rouge_scores":
                        results.update(res)
                    else:
                        results[metric_name] = round(res, 4)
                except Exception as exc:
                    logger.warning(f"Metric {metric_name} generated an exception: {exc}")
    else:
        if reference:
            results["exact_match"] = round(exact_match(answer, reference), 4)
            results["bert_score"] = get_bert_score(answer, reference)
            results.update(get_rouge_scores(answer, reference))
        
        if context:
            results["faithfulness"] = round(faithfulness(answer, context), 4)
            
        results["relevance"] = round(relevance(answer, question), 4)
        
    scores = [v for k,v in results.items() if isinstance(v, (int, float))]
    results["overall"] = round(sum(scores) / len(scores), 4) if scores else 0.0
    
    end_time = time.perf_counter()
    # Explicitly calculate and track latency for the evaluated metrics request
    results["eval_time_ms"] = round((end_time - start_time) * 1000)
    logger.info(f"eval_metrics parallel evaluation completed in {results['eval_time_ms']} ms")
    
    return results
