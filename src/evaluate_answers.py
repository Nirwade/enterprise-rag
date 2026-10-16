from rag import generate_answer

tests = [
    (
        "What should I do if I suspect a security incident?",
        ["security@acme.example", "one hour"],
    ),
    (
        "How much paid time off do full-time employees receive?",
        ["20 days"],
    ),
    (
        "When must business travel expenses be submitted?",
        ["30 days"],
    ),
]

for question, expected_facts in tests:
    answer, _ = generate_answer(question)
    answer_lower = answer.lower()

    passed = all(fact.lower() in answer_lower for fact in expected_facts)

    print(f"\n{'PASS' if passed else 'CHECK'}: {question}")
    print(answer)

print("\nEvaluation complete.")
