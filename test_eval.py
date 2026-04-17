import os
import sys
from services.llm import llm_service

test_cases = [
    {
        "name": "Test Case 1 — Same Proof, Two Valid Methods",
        "question": "Prove that: If n^2 is even, then n is even.",
        "teacher": "We prove the statement by contraposition. Contrapositive: If n is odd, then n^2 is odd. Let n=2k+1, where k is an integer. Then: n^2=(2k+1)^2=4k^2+4k+1=2(2k^2+2k)+1. Thus n^2 is odd. Since the contrapositive is true, the original statement is true.",
        "student": "Assume n^2 is even but n is odd. If n is odd, let n=2k+1. Then n^2=(2k+1)^2=4k^2+4k+1, which is odd. This contradicts the assumption that n^2 is even. Therefore, n must be even."
    },
    {
        "name": "Test Case 2a — Partial Correctness",
        "question": "Explain the difference between TCP and UDP.",
        "teacher": "TCP is a connection-oriented protocol that provides reliable, ordered, and error-checked delivery of data. It includes congestion control and flow control mechanisms. UDP is a connectionless protocol that provides faster but unreliable data transmission without guaranteeing delivery, order, or duplicate protection.",
        "student": "TCP is reliable and UDP is faster."
    },
    {
        "name": "Test Case 2b — Full Correctness",
        "question": "Explain the difference between TCP and UDP.",
        "teacher": "TCP is a connection-oriented protocol that provides reliable, ordered, and error-checked delivery of data. It includes congestion control and flow control mechanisms. UDP is a connectionless protocol that provides faster but unreliable data transmission without guaranteeing delivery, order, or duplicate protection.",
        "student": "TCP provides reliable, connection-oriented communication with error checking and congestion control, whereas UDP is connectionless and does not guarantee delivery or order, but has lower overhead and latency."
    },
    {
        "name": "Test Case 3a — Integration Correctness (Correct)",
        "question": "Evaluate: ∫2x dx",
        "teacher": "∫2x dx = x^2 + C",
        "student": "x^2 + C"
    },
    {
        "name": "Test Case 3b — Integration Correctness (Incorrect)",
        "question": "Evaluate: ∫2x dx",
        "teacher": "∫2x dx = x^2 + C",
        "student": "2x^2 + C"
    },
    {
        "name": "Test Case 3c — Integration Correctness (Partially Correct)",
        "question": "Evaluate: ∫2x dx",
        "teacher": "∫2x dx = x^2 + C",
        "student": "x^2"
    }
]

for tc in test_cases:
    print(f"\n==================================================")
    print(f"Running: {tc['name']}")
    result = llm_service.evaluate_answer(
        question=tc['question'],
        student_answer=tc['student'],
        reference_context=tc['teacher']
    )
    print(result)
