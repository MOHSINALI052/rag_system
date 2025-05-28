import re
from typing import List, Dict

def chunk_text(paragraphs: List[str], max_length: int = 500) -> List[Dict]:
    chunks = []
    for i, para in enumerate(paragraphs):
        para = re.sub(r'\s+', ' ', para).strip()
        sentences = re.split(r'(?<=[.!?])\s+', para)
        chunk = ""
        for sentence in sentences:
            if len(chunk) + len(sentence) <= max_length:
                chunk += sentence + " "
            else:
                chunks.append({"text": chunk.strip(), "source": i})
                chunk = sentence + " "
        if chunk:
            chunks.append({"text": chunk.strip(), "source": i})
    return chunks