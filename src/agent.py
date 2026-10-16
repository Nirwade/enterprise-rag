import json
import sys
import requests

from semantic_search import search, load_chunks

CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "llama3.1:8b"

def search_policies(query: str) -> str:
    results = search(query, top_k=3)
    return json.dumps([
        {"section": chunk["title"], "text": chunk["text"]}
        for _, chunk in results
    ])

def list_policy_sections() -> str:
    return json.dumps([chunk["title"] for chunk in load_chunks()])

tools = [
    {
        "type": "function",
        "function": {
            "name": "search_policies",
            "description": "Search company policies for an answer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_policy_sections",
            "description": "List all available company policy sections.",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]

available_tools = {
    "search_policies": search_policies,
    "list_policy_sections": list_policy_sections,
}

def run_agent(question):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a company policy assistant. "
                "Use the tools to retrieve company information. "
                "Answer using retrieved evidence and cite the section. "
                "If the documents lack the answer, say so."
            )
        },
        {"role": "user", "content": question}
    ]

    for _ in range(5):  # Prevent an endless agent loop.
        response = requests.post(
            CHAT_URL,
            json={
                "model": MODEL,
                "messages": messages,
                "tools": tools,
                "stream": False,
                "options": {"temperature": 0}
            },
            timeout=180
        )
        response.raise_for_status()
        message = response.json()["message"]
        messages.append(message)

        calls = message.get("tool_calls", [])

        if not calls:
            print("\nANSWER:")
            print(message.get("content", "No answer returned."))
            return

        for call in calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]

            print(f"TOOL CALLED: {name}")

            if name not in available_tools:
                result = "Error: unknown tool."
            else:
                try:
                    result = available_tools[name](**args)
                except Exception as exc:
                    result = f"Tool failed: {exc}"

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": str(result)
            })

    print("Stopped: maximum tool-call rounds reached.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/agent.py "Your question"')
        sys.exit(1)

    run_agent(" ".join(sys.argv[1:]))
