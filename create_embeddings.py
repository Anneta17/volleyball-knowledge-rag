import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


project_dir = Path(__file__).resolve().parent
chunks_path = project_dir / "data" / "processed" / "chunks.json"
embeddings_path = project_dir / "data" / "processed" / "embeddings.npy"

with chunks_path.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

texts = [chunk["text"] for chunk in chunks]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)

np.save(embeddings_path, embeddings)

print(f"Chunks: {len(chunks)}")
print(f"Σχήμα embeddings: {embeddings.shape}")
print(f"Αποθηκεύτηκαν στο: {embeddings_path}")