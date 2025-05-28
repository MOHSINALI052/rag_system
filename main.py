from pdf_loader import load_pdfs
from text_processor import chunk_text
from embeddings import embed_chunks
from vector_db import create_vector_store
from query_engine import retrieve_context, ask_hf_llm
from sentence_transformers import SentenceTransformer

pdf_path = "data_source"
pdf_list = [
    "BMF_2013_07_24.pdf",
    "BMF_2014_01_13_Änderung_von_2013_07_24.pdf",
    "BMF_2017_12_06.pdf",
    "BMF_2017_12_21.pdf",
    "BMF_2019_08_08_Änderung_von_2017_12_06.pdf",
    "BMF_2020_02_17_Änderung_von_2017_12_21.pdf",
    "BMF_2022_02_11_Änderung_von_2017_12_21_und_2020_02_17.pdf",
    "BMF_2022_03_18_Änderung_von_2021_08_21.pdf",
    "BMF_2023_10_05.pdf"
]

# Step 1: Load PDFs
raw_data = load_pdfs(pdf_path, pdf_list)

# Step 2: Chunk Text
all_chunks = []
for fname, paras in raw_data.items():
    for chunk in chunk_text(paras):
        all_chunks.append({"filename": fname, "text": chunk["text"], "source": chunk["source"]})

# Step 3: Generate Embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")
embedded_chunks = embed_chunks(all_chunks, model)

# Step 4: Store in Vector DB
index, db_chunks = create_vector_store(embedded_chunks)

# Step 5: Query Answering
questions = [
    "What is the basic allowance?",
    "How are pension benefits from a direct commitment or support fund treated for tax purposes?",
    "How are benefits from a direct insurance, pension fund, or pension scheme taxed during the payout phase?"
]

for q in questions:
    context = retrieve_context(q, model, index, db_chunks)
    answer = ask_hf_llm(q, context)
    print(f"\nQ: {q}\nA: {answer}")