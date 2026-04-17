import json
from services.generator_agent import generator_agent

context = """
The mitochondria is a double-membrane-bound organelle found in most eukaryotic organisms. Mitochondria generate most of the cell's supply of adenosine triphosphate (ATP), used as a source of chemical energy. A mitochondrion is thus termed the powerhouse of the cell.
"""

def test_agentic_generation():
    print("Testing Generation & Critique Agent...")
    questions = generator_agent.generate_with_critique(
        context=context,
        topics=["Mitochondria"],
        difficulty="Hard",
        num_questions=3,
        type_instruction="Generate only Multiple Choice Questions (MCQ)."
    )
    
    print("\n--- Final Results ---")
    print(json.dumps(questions, indent=2))
    
    assert isinstance(questions, list), "Output must be a list."
    assert len(questions) == 3, f"Expected 3 questions, got {len(questions)}."
    
    print("\n✅ Verification Passed!")

if __name__ == "__main__":
    test_agentic_generation()
