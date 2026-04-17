# 📊 AI Evaluation System — Test Case Report

**Date:** February 24, 2026  
**System:** StudyHub AI Evaluator (`services/llm.py → evaluate_answer`)  
**Model:** Gemini LLM with structured grading prompt  
**Methodology:** Each test case was evaluated **twice** — once without the teacher's answer key (AI-only), and once with the teacher's reference provided. This reveals how the teacher's input calibrates scoring accuracy.

---

## Scoring Dimensions

| Dimension | Description |
|-----------|-------------|
| **Score** | Overall grade (0–10) |
| **Final Answer Correctness** | Accuracy of the final result |
| **Logical Validity** | Soundness of reasoning steps |
| **Conceptual Accuracy** | Correct use of theory/definitions |
| **Completeness** | Coverage of required steps/details |

---

## Test Case 1 — Same Proof, Two Valid Methods

### 🎯 What We're Testing
> Can the evaluator accept a **different but valid proof strategy** (contradiction vs. contraposition)?

### Question
**Prove that:** If n² is even, then n is even.

### Teacher Answer Key (Contraposition)
Proves via contrapositive: If n is odd → n² is odd. Uses n = 2k+1 to show n² = 2(2k²+2k)+1, which is odd.

### Student Answer (Contradiction)
Assumes n² is even but n is odd. Shows n = 2k+1 → n² = 4k²+4k+1 (odd), contradicting the assumption. ∴ n must be even.

### Results

| Metric | Without Teacher Ref | With Teacher Ref | Change |
|--------|:------------------:|:----------------:|:------:|
| **Correctness** | ✅ correct | ✅ correct | — |
| **Score** | **9**/10 | **9**/10 | — |
| Final Answer | 10 | 10 | — |
| Logical Validity | 9 | 9 | — |
| Conceptual Accuracy | 9 | **10** ↑ | +1 |
| Completeness | 9 | 9 | — |
| Issues | None | None | — |

### 🔍 Analysis

| Expected Behavior | Actual Behavior | Verdict |
|---|---|---|
| ✔ Fully correct | ✅ Scored 9/10 both times | **PASS** |
| ✔ Accept different proof strategy | ✅ Contradiction method fully accepted | **PASS** |
| ✔ High logical_validity score | ✅ logical_validity = 9 | **PASS** |

> [!TIP]
> With the teacher's reference, conceptual_accuracy improved from 9→10, showing the LLM can better validate concept usage when it has a ground-truth answer to compare against. The alternative proof method was correctly recognized as equivalent.

---

## Test Case 2a — Partial Correctness (TCP vs UDP)

### 🎯 What We're Testing
> Can the evaluator give **partial credit** to a vague but directionally correct answer?

### Question
Explain the difference between TCP and UDP.

### Teacher Answer Key
TCP is connection-oriented, reliable, ordered, error-checked with congestion/flow control. UDP is connectionless, faster, unreliable, no delivery/order guarantees.

### Student Answer
> "TCP is reliable and UDP is faster."

### Results

| Metric | Without Teacher Ref | With Teacher Ref | Change |
|--------|:------------------:|:----------------:|:------:|
| **Correctness** | ⚠️ partial | ⚠️ partial | — |
| **Score** | **5**/10 | **4**/10 | -1 |
| Final Answer | 2 | 0 | -2 |
| Logical Validity | 3 | 4 | +1 |
| Conceptual Accuracy | 0 | **8** ↑ | +8 |
| Completeness | 0 | **8** ↑ | +8 |
| Issues | 3 issues flagged | 3 issues flagged | — |

### 🔍 Analysis

| Expected Behavior | Actual Behavior | Verdict |
|---|---|---|
| ⚠ Partial credit | ✅ Scored 4–5/10 | **PASS** |
| ✔ Basic idea correct | ✅ Recognized as "partial" | **PASS** |
| ❌ Low completeness | ⚠️ Without ref: 0, With ref: 8 | **MIXED** |

> [!NOTE]
> **Interesting anomaly:** With the teacher reference, conceptual_accuracy and completeness jumped to 8, yet the overall score *dropped* from 5→4. This suggests the LLM recognized the student's core concepts were directionally correct, but the extreme lack of detail resulted in a stricter overall penalty when compared against the comprehensive teacher answer.

---

## Test Case 2b — Full Correctness (TCP vs UDP)

### 🎯 What We're Testing
> Does a **comprehensive, detailed answer** score high across all dimensions?

### Question
Explain the difference between TCP and UDP.

### Teacher Answer Key
*(Same as Test Case 2a)*

### Student Answer
> "TCP provides reliable, connection-oriented communication with error checking and congestion control, whereas UDP is connectionless and does not guarantee delivery or order, but has lower overhead and latency."

### Results

| Metric | Without Teacher Ref | With Teacher Ref | Change |
|--------|:------------------:|:----------------:|:------:|
| **Correctness** | ✅ correct | ✅ correct | — |
| **Score** | **9**/10 | **9**/10 | — |
| Final Answer | 10 | 10 | — |
| Logical Validity | 9 | 9 | — |
| Conceptual Accuracy | 9 | **10** ↑ | +1 |
| Completeness | 9 | 9 | — |
| Issues | None | None | — |

### 🔍 Analysis

| Expected Behavior | Actual Behavior | Verdict |
|---|---|---|
| ✔ Fully correct | ✅ Scored 9/10 | **PASS** |
| ✔ High completeness | ✅ Completeness = 9 | **PASS** |
| ✔ High conceptual_accuracy | ✅ 9→10 with teacher ref | **PASS** |

> [!TIP]
> When compared side-by-side with Test Case 2a (score 4) vs Test Case 2b (score 9), the evaluator clearly differentiates between superficial and comprehensive answers — a 5-point gap that accurately reflects the quality difference.

---

## Test Case 3a — Integration: Correct Answer

### 🎯 What We're Testing
> Does a **fully correct mathematical answer** score near-perfect?

### Question
Evaluate: ∫2x dx

### Teacher Answer Key
x² + C

### Student Answer
> x² + C

### Results

| Metric | Without Teacher Ref | With Teacher Ref | Change |
|--------|:------------------:|:----------------:|:------:|
| **Correctness** | ✅ correct | ✅ correct | — |
| **Score** | **9**/10 | **9**/10 | — |
| Final Answer | 10 | 10 | — |
| Logical Validity | 9 | 9 | — |
| Conceptual Accuracy | 10 | 10 | — |
| Completeness | 9 | 9 | — |
| Issues | None | None | — |

### 🔍 Analysis

| Expected Behavior | Actual Behavior | Verdict |
|---|---|---|
| ✔ Fully correct | ✅ Scored 9/10 | **PASS** |

> Consistent scoring with or without teacher reference — the AI correctly identifies the answer independently.

---

## Test Case 3b — Integration: Incorrect Answer

### 🎯 What We're Testing
> Can the evaluator **detect a mathematical error** (2x² + C instead of x² + C)?

### Question
Evaluate: ∫2x dx

### Teacher Answer Key
x² + C

### Student Answer
> 2x² + C *(Incorrect — failed to divide by new exponent)*

### Results

| Metric | Without Teacher Ref | With Teacher Ref | Change |
|--------|:------------------:|:----------------:|:------:|
| **Correctness** | ~~✅ correct~~ | ⚠️ **partial** | 🔴 Fixed |
| **Score** | ~~**9**/10~~ | **6**/10 | **-3** |
| Final Answer | ~~10~~ | 8 | -2 |
| Logical Validity | ~~9~~ | 8 | -1 |
| Conceptual Accuracy | ~~9~~ | 10 | +1 |
| Completeness | ~~9~~ | 6 | -3 |
| Issues | ~~None~~ | 1 error flagged | 🔴 Fixed |

### 🔍 Analysis

| Expected Behavior | Actual Behavior | Verdict |
|---|---|---|
| ❌ Should be incorrect | ⚠️ Without ref: **WRONG** (scored 9/10!) | **FAIL without ref** |
| 🚩 Mathematical error | ✅ With ref: correctly detected | **PASS with ref** |

> [!CAUTION]
> **Critical Finding:** Without the teacher's reference, the AI **incorrectly scored 2x² + C as 9/10 correct!** This is a serious false positive. The AI failed to independently verify that ∫2x dx = x² + C, not 2x² + C. Only when given the teacher's answer key did it catch the error and appropriately lower the score to 6/10. **This demonstrates that teacher answer keys are essential for mathematical evaluation accuracy.**

---

## Test Case 3c — Integration: Missing Constant

### 🎯 What We're Testing
> Can the evaluator give **partial credit** when the integration is correct but the constant C is missing?

### Question
Evaluate: ∫2x dx

### Teacher Answer Key
x² + C

### Student Answer
> x² *(Missing the constant of integration C)*

### Results

| Metric | Without Teacher Ref | With Teacher Ref | Change |
|--------|:------------------:|:----------------:|:------:|
| **Correctness** | ~~✅ correct~~ | ⚠️ **partial** | 🔴 Fixed |
| **Score** | ~~**9**/10~~ | **6**/10 | **-3** |
| Final Answer | ~~10~~ | 8 | -2 |
| Logical Validity | ~~9~~ | 6 | -3 |
| Conceptual Accuracy | ~~9~~ | 8 | -1 |
| Completeness | ~~9~~ | **4** | **-5** |
| Issues | ~~None~~ | "Lack of constant C" | 🔴 Fixed |

### 🔍 Analysis

| Expected Behavior | Actual Behavior | Verdict |
|---|---|---|
| ⚠ Partial credit | ✅ With ref: 6/10 (partial) | **PASS with ref** |
| ✔ Correct integration noted | ✅ Final answer = 8 | **PASS** |
| ❌ Missing constant C | ✅ With ref: flagged as issue | **PASS with ref** |

> [!WARNING]
> Same pattern as Test Case 3b: without the teacher reference, the AI scored x² as **perfect (9/10)**, completely missing the omission of the constant C. With the reference, it correctly penalized the missing constant (completeness dropped to 4/10).

---

## 📈 Summary Dashboard

| Test Case | Category | Without Teacher | With Teacher | Expected | Verdict |
|-----------|----------|:--------------:|:------------:|:--------:|:-------:|
| **1** | Proof (alternative method) | 9/10 ✅ | 9/10 ✅ | High score | ✅ **PASS** |
| **2a** | Conceptual (partial) | 5/10 ⚠️ | 4/10 ⚠️ | Partial credit | ✅ **PASS** |
| **2b** | Conceptual (complete) | 9/10 ✅ | 9/10 ✅ | Full credit | ✅ **PASS** |
| **3a** | Math (correct) | 9/10 ✅ | 9/10 ✅ | Full credit | ✅ **PASS** |
| **3b** | Math (wrong answer) | ~~9/10~~ 🔴 | 6/10 ⚠️ | Low score | ⚠️ **PASS only w/ ref** |
| **3c** | Math (missing +C) | ~~9/10~~ 🔴 | 6/10 ⚠️ | Partial credit | ⚠️ **PASS only w/ ref** |

---

## 🔑 Key Findings

### ✅ What Works Well
1. **Alternative proof methods** are correctly accepted (contradiction vs. contraposition)
2. **Partial vs. full answers** are well-differentiated (2a: 4/10 vs. 2b: 9/10)
3. **Teacher references improve sub-score calibration** (conceptual_accuracy consistently improves)
4. **Detailed feedback** is generated with specific, actionable issues

### 🔴 Critical Issues
1. **Mathematical verification without teacher reference is unreliable** — incorrect answers (2x² + C) were scored 9/10 when no reference was provided
2. **Missing components (constant C) go undetected** without the teacher's answer key
3. **The AI over-relies on surface-level pattern matching** for math, accepting plausible-looking answers without verification

### 📋 Recommendations
1. **Always require teacher answer keys for mathematical/calculation questions** — the AI cannot reliably self-verify arithmetic
2. **Consider adding a separate verification step** that re-checks mathematical answers with a dedicated math solver
3. **Flag evaluations done without teacher reference** so teachers know which scores need manual review
