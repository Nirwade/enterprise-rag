import math
import re
import sys
from collections import Counter
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:11434/api/embed"
EMBED_MODEL = "nomic-embed-text"
PROJECT_DIR = Path(__file__).resolve().parents[1]
DOCUMENT_PATH = PROJECT_DIR / "documents" / "employee_handbook.md"


def embed(text):
    response = requests.post(
        OLLAMA_URL,
        json={"model": EMBED_MODEL, "input": text},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["embeddings"][0]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def tokenize(text):
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [
        w[:-3] + "y" if w.endswith("ies") and len(w) > 4
        else w[:-1] if w.endswith("s") and len(w) > 3
        and not w.endswith(("ss", "us", "is"))
        else w
        for w in words
    ]


def load_chunks():
    document = DOCUMENT_PATH.read_text(encoding="utf-8")
    sections = re.split(r"(?m)(?=^##\s+)", document)
    chunks = []

    for section in sections:
        section = section.strip()
        if section.startswith("## ") and len(section.split()) >= 5:
            title = section.splitlines()[0].replace("## ", "", 1)
            chunks.append({"title": title, "text": section})

    return chunks


def keyword_scores(question, chunks):
    docs = [tokenize(c["text"]) for c in chunks]
    query = set(tokenize(question))
    n = len(docs)
    avg_len = sum(map(len, docs)) / max(n, 1)
    document_frequency = Counter(
        word for doc in docs for word in set(doc)
    )
    scores = []

    for doc in docs:
        frequencies = Counter(doc)
        score = 0.0
        for word in query:
            freq = frequencies[word]
            if freq:
                df = document_frequency[word]
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                denominator = freq + 1.5 * (
                    0.25 + 0.75 * len(doc) / max(avg_len, 1)
                )
                score += idf * freq * 2.5 / denominator
        scores.append(score)

    return scores


def search(question, top_k=3):
    chunks = load_chunks()
    if not chunks:
        raise ValueError(f"No sections found in {DOCUMENT_PATH}")

    query_vector = embed(question)
    vectors = [embed(c["text"]) for c in chunks]

    semantic = [
        cosine_similarity(query_vector, vector) for vector in vectors
    ]
    lexical = keyword_scores(question, chunks)

    semantic_order = sorted(
        range(len(chunks)), key=lambda i: semantic[i], reverse=True
    )
    lexical_order = sorted(
        range(len(chunks)), key=lambda i: lexical[i], reverse=True
    )

    # Reciprocal Rank Fusion combines both rankings.
    combined = {}
    for rank, i in enumerate(semantic_order, 1):
        combined[i] = combined.get(i, 0) + 1 / (60 + rank)
    for rank, i in enumerate(lexical_order, 1):
        combined[i] = combined.get(i, 0) + 1 / (60 + rank)

    ranked = sorted(combined, key=combined.get, reverse=True)
    return [(combined[i], chunks[i]) for i in ranked[:top_k]]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/semantic_search.py "Your question"')
        sys.exit(1)

    for score, chunk in search(" ".join(sys.argv[1:])):
        print(f"\n{chunk['title']} | Hybrid score: {score:.4f}")
        print(chunk["text"])
