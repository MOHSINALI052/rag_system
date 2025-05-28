from sentence_transformers import SentenceTransformer
from typing import List, Dict

def embed_chunks(chunks: List[Dict], model: SentenceTransformer) -> List[Dict]:
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb
    return chunks