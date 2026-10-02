from pathlib import Path
from pypdf import PdfReader


pdf_dir = Path(__file__).resolve().parent / "data" / "Pdfs"
pages = []

for pdf_path in sorted(pdf_dir.glob("*.pdf")):
    reader = PdfReader(pdf_path)
    pages_from_this_pdf = 0

    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()

        if not text:
            continue

        pages.append({
            "source": pdf_path.name,
            "page": page_number,
            "text": text,
        })
        pages_from_this_pdf += 1

    print(f"{pdf_path.name}: {pages_from_this_pdf} σελίδες με κείμενο")

print(f"\nΣύνολο: {len(pages)} σελίδες με κείμενο")

# Δείγμα: η σελίδα 44 του PDF με τους κανόνες
example = next(
    (
        item for item in pages
        if "Rules2025_2028" in item["source"] and item["page"] == 44
    ),
    None,
)

if example:
    print("\nΠαράδειγμα εγγραφής:")
    print(f"Πηγή: {example['source']}")
    print(f"Σελίδα PDF: {example['page']}")
    print(f"Κείμενο: {example['text'][:700]}")