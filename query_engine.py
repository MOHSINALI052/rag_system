from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import requests
import json
from config import HuggingFaceConfig

def retrieve_context(query: str, model: SentenceTransformer, index, data: List[dict], k: int = 3) -> str:
    query_vec = model.encode([query])[0]
    D, I = index.search(np.array([query_vec]), k)
    return "\n\n".join([data[i]['text'] for i in I[0]])

def ask_hf_llm(query: str, context: str) -> str:
    prompt = f"""You are a helpful assistant. Answer based on the context below:

Context:
{context}

Question:
{query}

Answer:"""

    headers = {
        "Authorization": f"Bearer {HuggingFaceConfig.API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 300, "temperature": 0.5}
    }
    res = requests.post("https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta",
                        headers=headers, data=json.dumps(data))
    if res.status_code == 200:
        return res.json()[0]["generated_text"].split("Answer:")[-1].strip()
    else:
        return f"Error: {res.status_code}, {res.text}"