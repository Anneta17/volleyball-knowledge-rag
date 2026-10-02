import json
from pathlib import Path

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter


project_dir = Path(__file__).resolve().parent
pdf_dir = project_dir / "data" / "Pdfs"
output_path = project_dir / "data" / "processed" / "chunks.json"

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
)

chunks = []

for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    reader = PdfReader(pdf_path)

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            continue

        page_chunks = splitter.split_text(text)

        for chunk_number, chunk_text in enumerate(page_chunks, start=1):
            chunks.append({
                "text": chunk_text,
                "source": pdf_path.name,
                "page": page_number,
                "chunk_number": chunk_number,
            })

output_path.parent.mkdir(parents=True, exist_ok=True)

with output_path.open("w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

print(f"Αποθηκεύτηκαν {len(chunks)} chunks στο:")
print(output_path)