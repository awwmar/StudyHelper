"""
Evaluation Report — Parallel Version

Runs all test cases concurrently using ThreadPoolExecutor.
Evaluates with and without teacher reference in parallel.

Usage:
    cd c:\Users\Amar\Desktop\SOP
    python run_eval_report.py
"""
import os
import sys
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

from services.llm import llm_service

test_cases = [
    {
        "name": "Test Case 1 — Same Proof, Two Valid Methods",
        "question": "Prove that: If n^2 is even, then n is even.",
        "teacher": "We prove the statement by contraposition. Contrapositive: If n is odd, then n^2 is odd. Let n=2k+1, where k is an integer. Then: n^2=(2k+1)^2=4k^2+4k+1=2(2k^2+2k)+1. Thus n^2 is odd. Since the contrapositive is true, the original statement is true.",
        "student": "Assume n^2 is even but n is odd. If n is odd, let n=2k+1. Then n^2=(2k+1)^2=4k^2+4k+1, which is odd. This contradicts the assumption that n^2 is even. Therefore, n must be even."
    },
    {
        "name": "Test Case 2a — Partial Correctness (TCP/UDP)",
        "question": "Explain the difference between TCP and UDP.",
        "teacher": "TCP is a connection-oriented protocol that provides reliable, ordered, and error-checked delivery of data. It includes congestion control and flow control mechanisms. UDP is a connectionless protocol that provides faster but unreliable data transmission without guaranteeing delivery, order, or duplicate protection.",
        "student": "TCP is reliable and UDP is faster."
    },
    {
        "name": "Test Case 2b — Full Correctness (TCP/UDP)",
        "question": "Explain the difference between TCP and UDP.",
        "teacher": "TCP is a connection-oriented protocol that provides reliable, ordered, and error-checked delivery of data. It includes congestion control and flow control mechanisms. UDP is a connectionless protocol that provides faster but unreliable data transmission without guaranteeing delivery, order, or duplicate protection.",
        "student": "TCP provides reliable, connection-oriented communication with error checking and congestion control, whereas UDP is connectionless and does not guarantee delivery or order, but has lower overhead and latency."
    },
    {
        "name": "Test Case 3a — Integration Correct",
        "question": "Evaluate: integral of 2x dx",
        "teacher": "integral of 2x dx = x^2 + C",
        "student": "x^2 + C"
    },
    {
        "name": "Test Case 3b — Integration Incorrect",
        "question": "Evaluate: integral of 2x dx",
        "teacher": "integral of 2x dx = x^2 + C",
        "student": "2x^2 + C"
    },
    {
        "name": "Test Case 3c — Integration Partially Correct",
        "question": "Evaluate: integral of 2x dx",
        "teacher": "integral of 2x dx = x^2 + C",
        "student": "x^2"
    }
]


def parse_eval(raw):
    try:
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
        return json.loads(raw)
    except:
        return {"raw": raw}


def evaluate_test_case(tc):
    """Evaluate a single test case (both with and without teacher reference)."""
    result_no_ref = llm_service.evaluate_answer(
        question=tc['question'],
        student_answer=tc['student']
    )
    result_with_ref = llm_service.evaluate_answer(
        question=tc['question'],
        student_answer=tc['student'],
        reference_context=tc['teacher']
    )
    return {
        "name": tc['name'],
        "question": tc['question'],
        "teacher_answer": tc['teacher'],
        "student_answer": tc['student'],
        "eval_without_teacher": parse_eval(result_no_ref),
        "eval_with_teacher": parse_eval(result_with_ref)
    }


def main():
    print(f"Running {len(test_cases)} test cases in parallel...\n")
    start_time = time.time()

    results = [None] * len(test_cases)

    # Run all test cases concurrently
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_idx = {
            executor.submit(evaluate_test_case, tc): i
            for i, tc in enumerate(test_cases)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
                print(f"  ✅ Completed: {test_cases[idx]['name']}")
            except Exception as e:
                print(f"  ❌ Failed: {test_cases[idx]['name']} — {e}")
                results[idx] = {
                    "name": test_cases[idx]['name'],
                    "error": str(e)
                }

    elapsed = time.time() - start_time

    # Print results
    for entry in results:
        if not entry or 'error' in entry:
            continue
        print(f"\n{'='*60}")
        print(f"{entry['name']}")
        print(f"{'='*60}")
        print(f"\n--- Without Teacher Reference ---")
        print(json.dumps(entry['eval_without_teacher'], indent=2))
        print(f"\n--- With Teacher Reference ---")
        print(json.dumps(entry['eval_with_teacher'], indent=2))

    # Save full results to JSON
    with open("eval_full_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n\nResults saved to eval_full_results.json")
    print(f"Total time: {elapsed:.1f}s (parallel) vs ~{elapsed * len(test_cases) / max(1, sum(1 for r in results if r)):.0f}s estimated sequential")


if __name__ == "__main__":
    main()
