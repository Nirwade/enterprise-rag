from semantic_search import search
from reranker import rerank

tests = [
    ("What email should I contact for a security incident?", "Security Policy"),
    ("How many paid vacation days do employees get?", "Leave Policy"),
    ("When must travel expenses be submitted?", "Expense Reimbursement"),
    ("Who approves fully remote work?", "Remote Work Policy"),
    ("What are the IT support hours?", "IT Support"),
]

for question, expected in tests:
    before = [c["title"] for _, c in search(question, top_k=5)]
    after = [c["title"] for c in rerank(question)]

    old_rank = before.index(expected) + 1 if expected in before else None
    new_rank = after.index(expected) + 1 if expected in after else None

    print(f"\n{expected}: before={old_rank}, after={new_rank}")

print("\nEvaluation complete.")
