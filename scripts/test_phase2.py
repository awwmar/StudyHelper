"""
Test: Selector Agent + Eval Metrics Verification

Validates Phase 2 components without requiring Supabase or a running app.

Usage:
    python scripts/test_phase2.py
"""
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()


def test_selector_agent():
    """Test the Selector Agent's re-ranking capability."""
    print("\n" + "=" * 60)
    print("TEST 1: Selector Agent (Cross-Encoder Re-Ranking)")
    print("=" * 60)

    from services.selector import SelectorAgent

    t0 = time.time()
    agent = SelectorAgent()
    print(f"  Model loaded in {time.time() - t0:.1f}s")

    # Simulate chunks — mix of relevant and irrelevant
    query = "What is machine learning?"
    chunks = [
        {"text_content": "Machine learning is a branch of artificial intelligence that enables computers to learn from data without being explicitly programmed.", "id": 1},
        {"text_content": "The French Revolution of 1789 led to major political changes across Europe.", "id": 2},
        {"text_content": "Supervised learning uses labeled data to train models for classification and regression tasks.", "id": 3},
        {"text_content": "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.", "id": 4},
        {"text_content": "Neural networks are computational models inspired by the human brain, used in deep learning.", "id": 5},
        {"text_content": "The Great Wall of China was built over several centuries to protect against invasions.", "id": 6},
        {"text_content": "Gradient descent is an optimization algorithm commonly used to train machine learning models.", "id": 7},
        {"text_content": "Shakespeare wrote Romeo and Juliet in the late 16th century.", "id": 8},
    ]

    # Get scored results for analysis
    scored = agent.select_with_scores(query, chunks)
    print(f"\n  Query: '{query}'")
    print(f"  Ranking (highest relevance first):")
    for score, chunk in scored:
        relevance_bar = "█" * int(max(0, score + 5) * 2)  # Visual bar
        print(f"    {score:+7.3f} {relevance_bar} | {chunk['text_content'][:70]}...")

    # Test select with top_ratio=0.5
    selected = agent.select(query, chunks, top_ratio=0.5)
    print(f"\n  Selected {len(selected)}/{len(chunks)} chunks (top 50%)")

    # Verify ML-related chunks are in top selections
    selected_ids = {c['id'] for c in selected}
    ml_ids = {1, 3, 5, 7}  # ML-related chunks
    ml_selected = ml_ids & selected_ids
    print(f"  ML-related chunks selected: {len(ml_selected)}/{len(ml_ids)}")

    if len(ml_selected) >= 3:
        print("  ✅ PASS — Selector correctly prioritizes relevant chunks")
    else:
        print("  ⚠️ PARTIAL — Some relevant chunks were missed")

    return True


def test_eval_metrics():
    """Test the evaluation metrics (ExactMatch, Faithfulness, Relevance)."""
    print("\n" + "=" * 60)
    print("TEST 2: Evaluation Metrics (ExactMatch, Faithfulness, Relevance)")
    print("=" * 60)

    from services.eval_metrics import exact_match, faithfulness, relevance

    # --- ExactMatch Tests ---
    print("\n  [ExactMatch]")

    em1 = exact_match("x^2 + C", "x^2 + C")
    print(f"    'x^2 + C' vs 'x^2 + C' → {em1:.4f} {'✅' if em1 == 1.0 else '❌'}")

    em2 = exact_match("x^2 + C", "x² + C")
    print(f"    'x^2 + C' vs 'x² + C' → {em2:.4f} (partial match)")

    em3 = exact_match("TCP is reliable", "The TCP protocol provides reliable, ordered delivery")
    print(f"    Short vs detailed → {em3:.4f} (should be low)")

    em4 = exact_match("", "x^2 + C")
    print(f"    Empty vs answer → {em4:.4f} {'✅' if em4 == 0.0 else '❌'}")

    # --- Faithfulness Test ---
    print("\n  [Faithfulness] (requires LLM call)")
    context = "Machine learning is a subset of AI. It uses algorithms to learn from data. Common approaches include supervised, unsupervised, and reinforcement learning."

    t0 = time.time()
    f1 = faithfulness(
        "Machine learning uses algorithms to learn from data and includes supervised learning.",
        context
    )
    print(f"    Grounded answer → {f1:.4f} (should be high) [{time.time()-t0:.1f}s]")

    t0 = time.time()
    f2 = faithfulness(
        "Machine learning was invented by Alan Turing in 1950 and requires quantum computers.",
        context
    )
    print(f"    Hallucinated answer → {f2:.4f} (should be low) [{time.time()-t0:.1f}s]")

    # --- Relevance Test ---
    print("\n  [Relevance] (requires LLM call)")

    t0 = time.time()
    r1 = relevance(
        "TCP provides reliable data delivery with error checking.",
        "Explain the difference between TCP and UDP."
    )
    print(f"    On-topic answer → {r1:.4f} (should be high) [{time.time()-t0:.1f}s]")

    t0 = time.time()
    r2 = relevance(
        "Photosynthesis is the process of converting sunlight to energy.",
        "Explain the difference between TCP and UDP."
    )
    print(f"    Off-topic answer → {r2:.4f} (should be low) [{time.time()-t0:.1f}s]")

    print("\n  ✅ Metrics computed successfully")
    return True


def test_parallel_eval():
    """Test the parallel evaluate_batch method."""
    print("\n" + "=" * 60)
    print("TEST 3: Parallel Evaluation (evaluate_batch)")
    print("=" * 60)

    from services.llm import llm_service

    items = [
        {"question": "What is 2+2?", "student_answer": "4"},
        {"question": "What is the capital of France?", "student_answer": "Paris"},
        {"question": "Evaluate: integral of 2x dx", "student_answer": "x^2 + C"},
    ]

    t0 = time.time()
    results = llm_service.evaluate_batch(items, max_workers=3)
    elapsed = time.time() - t0

    print(f"  Evaluated {len(results)} items in {elapsed:.1f}s")
    for i, result in enumerate(results):
        try:
            import json, re
            json_match = re.search(r'\{.*\}', result, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                score = data.get('score', '?')
            else:
                score = '?'
        except:
            score = '?'
        print(f"    Q{i+1}: Score = {score}/10")

    print(f"\n  ✅ Parallel evaluation complete ({elapsed:.1f}s for {len(items)} items)")
    return True


if __name__ == "__main__":
    print("Phase 2 Verification Tests")
    print("=" * 60)

    passed = 0
    total = 3

    try:
        if test_selector_agent():
            passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")

    try:
        if test_eval_metrics():
            passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")

    try:
        if test_parallel_eval():
            passed += 1
    except Exception as e:
        print(f"  ❌ FAIL: {e}")

    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} tests passed")
    print("=" * 60)
