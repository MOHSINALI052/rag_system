import os
import fitz  # PyMuPDF
from typing import List, Dict

def load_pdfs(pdf_dir: str, filenames: List[str]) -> Dict[str, List[str]]:
    texts = {}
    for filename in filenames:
        path = os.path.join(pdf_dir, filename)
        doc = fitz.open(path)
        paragraphs = []
        for page in doc:
            text = page.get_text().strip()
            if text:
                paragraphs.append(text)
        texts[filename] = paragraphs
    return texts