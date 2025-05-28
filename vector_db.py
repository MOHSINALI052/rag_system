import faiss
import numpy as np
from typing import List, Tuple, Dict

def create_vector_store(chunks: List[Dict]) -> Tuple[faiss.IndexFlatL2, List[Dict]]:
    dim = len(chunks[0]["embedding"])
    index = faiss.IndexFlatL2(dim)
    vectors = np.array([chunk["embedding"] for chunk in chunks])
    index.add(vectors)
    return index, chunks