import json
import requests
from semantic_search import search

def rerank(question):
    candidates = search(question, top_k=5)
    documents = [
        {"title": chunk["title"], "text": chunk["text"]}
        for _, chunk in candidates
    ]

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.1:8b",
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Rank documents by how directly they answer "
                        "the question. Return JSON with a 'ranking' "
                        "array containing document titles, most relevant first."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps({
                        "question": question,
                        "documents": documents,
                    }),
                },
            ],
        },
        timeout=180,
    )
    response.raise_for_status()
    result = json.loads(response.json()["message"]["content"])
    by_title = {doc["title"]: doc for doc in documents}
    ranked_titles = result.get("ranking", [])

    # Handle titles returned as strings or objects.
    titles = [
        item if isinstance(item, str) else item.get("title", "")
        for item in ranked_titles
    ]
    ranked = [by_title[t] for t in titles if t in by_title]
    seen = {doc["title"] for doc in ranked}
    ranked.extend(doc for doc in documents if doc["title"] not in seen)

    return ranked


if __name__ == "__main__":
    question = "Who do I contact if I think our systems have been compromised?"
    for rank, doc in enumerate(rerank(question), 1):
        print(f"{rank}. {doc['title']}")
