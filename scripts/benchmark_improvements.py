import time
import random
import json
from services.vector import vector_service
from services.selector import selector_agent
from services.llm import llm_service
from agents.system2_query import system2
from services.eval_metrics import evaluate_all
from config import config

# 20 Diverse queries for benchmark testing
TEST_QUERIES = [
    "What is cellular respiration?",
    "Explain the concept of Object-Oriented Programming.",
    "How does photosynthesis work?",
    "What are the Newton's Laws of Motion?",
    "Describe the water cycle.",
    "What is the theory of relativity?",
    "Explain how a CPU processes instructions.",
    "What are the basic principles of Quantum Mechanics?",
    "Describe the French Revolution.",
    "How does the immune system fight viruses?",
    "What is inflation in economics?",
    "Explain the structure of an atom.",
    "How do tectonic plates move?",
    "What is the Pythagorean theorem?",
    "Describe the process of mitosis.",
    "Explain the concept of supply and demand.",
    "What are the functions of the liver?",
    "How does a combustion engine work?",
    "What is climate change?",
    "Explain the basics of machine learning."
]

random.seed(42)

def run_pipeline(queries, use_new_pipeline):
    """
    Run pipeline on given queries and collect metrics.
    """
    results = {
        "retrieval_time": [],
        "selector_time": [],
        "llm_time": [],
        "eval_time": [],
        "e2e_time": [],
        "exact_match": [],
        "faithfulness": [],
        "bert_f1": [],
        "rougeL": [],
        "kg_chunks": []
    }
    
    # Store old configs
    old_hybrid = config.ENABLE_HYBRID_SEARCH
    old_kg = config.ENABLE_KNOWLEDGE_GRAPH
    old_filter = config.ENABLE_CHUNK_LENGTH_FILTER
    old_eval = config.ENABLE_PARALLEL_METRICS
    old_cache = config.ENABLE_LLM_CACHE
    
    # Set configs according to pipeline
    config.ENABLE_HYBRID_SEARCH = use_new_pipeline
    config.ENABLE_KNOWLEDGE_GRAPH = use_new_pipeline
    config.ENABLE_CHUNK_LENGTH_FILTER = use_new_pipeline
    config.ENABLE_PARALLEL_METRICS = use_new_pipeline
    config.ENABLE_LLM_CACHE = use_new_pipeline

    for query in queries:
        # End to end measurement
        t0_e2e = time.perf_counter()
        
        # 1. Retrieval
        t0_ret = time.perf_counter()
        chunks = system2.retriever_agent.retrieve(query, top_k=30)
        t1_ret = time.perf_counter()
        results["retrieval_time"].append(t1_ret - t0_ret)
        
        kg_count = 0
        if use_new_pipeline:
            # Approx check for kg chunks addition (if chunk ids match our KG mapping)
            # We don't distinctly separate them in retrieve, but we can measure count diff ideally
            # Here we just record default 0, will update based solely on KG run.
            pass
            
        # 2. Selector
        t0_sel = time.perf_counter()
        selected_chunks = system2.selector_agent.filter_chunks(query, chunks)
        t1_sel = time.perf_counter()
        results["selector_time"].append(t1_sel - t0_sel)
        
        context = "\n\n".join([c['text_content'] for c in selected_chunks])
        
        # 3. LLM Generation
        t0_llm = time.perf_counter()
        drafted = system2.response_agent.draft_output(context, ["General"], "Medium", 1, "Generate mixed questions.", "")
        t1_llm = time.perf_counter()
        results["llm_time"].append(t1_llm - t0_llm)
        
        # Parse drafted as mock system answer to run full metrics
        parsed_answer = drafted
        try:
            questions = json.loads(drafted)
            if questions:
                parsed_answer = questions[0]
        except:
            pass
            
        # 4. Evaluator (Extended Metrics)
        t0_eval = time.perf_counter()
        # Mock reference
        mock_reference = query + " Details." 
        eval_metrics = evaluate_all(parsed_answer, query, reference=mock_reference, context=context)
        t1_eval = time.perf_counter()
        results["eval_time"].append(t1_eval - t0_eval)
        
        # Record scores
        results["exact_match"].append(eval_metrics.get("exact_match", 0.0))
        results["faithfulness"].append(eval_metrics.get("faithfulness", 0.0))
        results["bert_f1"].append(eval_metrics.get("bert_score", 0.0))
        results["rougeL"].append(eval_metrics.get("rougeL", 0.0))
        
        t1_e2e = time.perf_counter()
        results["e2e_time"].append(t1_e2e - t0_e2e)
        
    # Restore old configs
    config.ENABLE_HYBRID_SEARCH = old_hybrid
    config.ENABLE_KNOWLEDGE_GRAPH = old_kg
    config.ENABLE_CHUNK_LENGTH_FILTER = old_filter
    config.ENABLE_PARALLEL_METRICS = old_eval
    config.ENABLE_LLM_CACHE = old_cache
    
    return results

def compute_stats(times):
    if not times:
        return 0, 0, 0
    t_sorted = sorted(times)
    mean = sum(t_sorted) / len(t_sorted)
    p50 = t_sorted[int(len(t_sorted) * 0.5)]
    p95 = t_sorted[int(len(t_sorted) * 0.95)]
    return mean * 1000, p50 * 1000, p95 * 1000  # ms

def baseline():
    print("Running Baseline...")
    return run_pipeline(TEST_QUERIES, use_new_pipeline=False)

def improved():
    print("Running Improved...")
    return run_pipeline(TEST_QUERIES, use_new_pipeline=True)

def print_comparison(base, imp):
    br_mean, _, _ = compute_stats(base["retrieval_time"])
    ir_mean, _, _ = compute_stats(imp["retrieval_time"])
    dr_r = ((ir_mean - br_mean) / max(br_mean, 1)) * 100
    
    bs_mean, _, _ = compute_stats(base["selector_time"])
    is_mean, _, _ = compute_stats(imp["selector_time"])
    ds_s = ((is_mean - bs_mean) / max(bs_mean, 1)) * 100
    
    bl_mean, _, _ = compute_stats(base["llm_time"])
    il_mean, _, _ = compute_stats(imp["llm_time"])
    dl_l = ((il_mean - bl_mean) / max(bl_mean, 1)) * 100

    be_mean, _, _ = compute_stats(base["eval_time"])
    ie_mean, _, _ = compute_stats(imp["eval_time"])
    de_e = ((ie_mean - be_mean) / max(be_mean, 1)) * 100

    _, _, bp95 = compute_stats(base["e2e_time"])
    _, _, ip95 = compute_stats(imp["e2e_time"])
    dp_p95 = ((ip95 - bp95) / max(bp95, 1)) * 100
    
    em_b_mean = sum(base["exact_match"]) / len(base["exact_match"]) if base["exact_match"] else 0
    em_i_mean = sum(imp["exact_match"]) / len(imp["exact_match"]) if imp["exact_match"] else 0
    d_em = em_i_mean - em_b_mean
    
    f_b_mean = sum(base["faithfulness"]) / len(base["faithfulness"]) if base["faithfulness"] else 0
    f_i_mean = sum(imp["faithfulness"]) / len(imp["faithfulness"]) if imp["faithfulness"] else 0
    d_f = f_i_mean - f_b_mean

    bert_i_mean = sum(imp["bert_f1"]) / len(imp["bert_f1"]) if imp["bert_f1"] else 0
    rouge_i_mean = sum(imp["rougeL"]) / len(imp["rougeL"]) if imp["rougeL"] else 0
    
    print("-" * 75)
    print(f"{'Metric':<25} | {'Baseline':<12} | {'Improved':<12} | {'Delta':<8}")
    print("-" * 75)
    print(f"{'Retrieval time (mean)':<25} | {br_mean:8.1f}ms   | {ir_mean:8.1f}ms   | {dr_r:+.1f}%")
    print(f"{'Selector time (mean)':<25} | {bs_mean:8.1f}ms   | {is_mean:8.1f}ms   | {ds_s:+.1f}%")
    print(f"{'LLM generation (mean)':<25} | {bl_mean:8.1f}ms   | {il_mean:8.1f}ms   | {dl_l:+.1f}%")
    print(f"{'Eval time (mean)':<25} | {be_mean:8.1f}ms   | {ie_mean:8.1f}ms   | {de_e:+.1f}%")
    print(f"{'End-to-end p95':<25} | {bp95:8.1f}ms   | {ip95:8.1f}ms   | {dp_p95:+.1f}%")
    print(f"{'ExactMatch':<25} | {em_b_mean:8.2f}       | {em_i_mean:8.2f}       | {d_em:+.2f}")
    print(f"{'Faithfulness':<25} | {f_b_mean:8.2f}       | {f_i_mean:8.2f}       | {d_f:+.2f}")
    print(f"{'BERTScore F1':<25} | {'N/A':<12} | {bert_i_mean:8.2f}       | NEW")
    print(f"{'ROUGE-L':<25} | {'N/A':<12} | {rouge_i_mean:8.2f}       | NEW")
    print(f"{'KG-augmented chunks':<25} | 0            | {'Not isolated':<12} | NEW")
    print("-" * 75)

if __name__ == "__main__":
    base_res = baseline()
    imp_res = improved()
    print_comparison(base_res, imp_res)
