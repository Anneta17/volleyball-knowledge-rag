import json
from pathlib import Path

import faiss
import numpy as np


project_dir = Path(__file__).resolve().parent
processed_dir = project_dir / "data" / "processed"

chunks_path = processed_dir / "chunks.json"
embeddings_path = processed_dir / "embeddings.npy"
index_path = processed_dir / "volleyball.faiss"

with chunks_path.open("r", encoding="utf-8") as file:
    chunks = json.load(file)

embeddings = np.load(embeddings_path)

if len(chunks) != len(embeddings):
    raise ValueError(
        "Ο αριθμός των chunks δεν ταιριάζει με τον αριθμό των embeddings."
    )

vectors = np.ascontiguousarray(embeddings, dtype="float32")
faiss.normalize_L2(vectors)

dimensions = vectors.shape[1]
index = faiss.IndexFlatIP(dimensions)
index.add(vectors)

faiss.write_index(index, str(index_path))

print(f"Διαστάσεις ανά vector: {dimensions}")
print(f"Vectors στον FAISS index: {index.ntotal}")
print(f"Αποθηκεύτηκε στο: {index_path}")