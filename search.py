import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


project_dir = Path(__file__).resolve().parent
processed_dir = project_dir / "data" / "processed"

chunks_path = processed_dir / "chunks.json"
index_path = processed_dir / "volleyball.faiss"

with chunks_path.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

index = faiss.read_index(str(index_path))

if index.ntotal != len(chunks):
    raise ValueError(
        "Ο FAISS index και το chunks.json έχουν διαφορετικό αριθμό εγγραφών."
    )

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

question = "What are the responsibilities of the libero?"

query_vector = model.encode(
    [question],
    normalize_embeddings=True,
)

query_vector = np.ascontiguousarray(query_vector, dtype="float32")

scores, positions = index.search(query_vector, k=5)

print(f"\nΕρώτηση: {question}")
print("\nthe top 5 most relative  chunks:")

for rank, (score, position) in enumerate(
    zip(scores[0], positions[0]),
    start=1,
):
    if position == -1:
        continue

    chunk = chunks[int(position)]

    print(f"\n{'=' * 60}")
    print(f"result #{rank} — score: {score:.3f}")
    print(f"source: {chunk['source']}")
    print(f"page PDF: {chunk['page']}")
    print(f"Chunk: {chunk['chunk_number']}")
    print(f"text: {chunk['text'][:500]}")