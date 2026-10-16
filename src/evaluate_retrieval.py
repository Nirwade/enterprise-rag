from semantic_search import search

tests = [
    ("What email should I contact for a security incident?", "Security Policy"),
    ("How many paid vacation days do employees get?", "Leave Policy"),
    ("When must travel expenses be submitted?", "Expense Reimbursement"),
    ("Who approves fully remote work?", "Remote Work Policy"),
    ("What are the IT support hours?", "IT Support"),
]

hit1 = 0
hit3 = 0
reciprocal_ranks = []

for question, expected in tests:
    results = search(question, top_k=100)

    rank = next(
        (i for i, (_, chunk) in enumerate(results, 1)
         if chunk["title"] == expected),
        None,
    )

    if rank is not None:
        hit1 += rank == 1
        hit3 += rank <= 3
        reciprocal_ranks.append(1 / rank)
    else:
        reciprocal_ranks.append(0)

    print(f"{'FOUND' if rank else 'MISSING'}: {expected} | Rank: {rank}")

n = len(tests)
print(f"\nHit@1: {hit1}/{n} ({hit1 / n:.0%})")
print(f"Hit@3: {hit3}/{n} ({hit3 / n:.0%})")
print(f"MRR:   {sum(reciprocal_ranks) / n:.3f}")
