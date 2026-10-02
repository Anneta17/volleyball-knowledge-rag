from pathlib import Path
from pypdf import PdfReader


# Ο φάκελος όπου βρίσκεται το inspect_pdfs.py
project_dir = Path(__file__).resolve().parent

# Ο φάκελος όπου έχεις βάλει τα PDF
pdf_dir = project_dir / "data" / "Pdfs"

print(f"Ψάχνω στον φάκελο: {pdf_dir}")

if not pdf_dir.is_dir():
    print("Ο φάκελος Pdfs δεν βρέθηκε.")
else:
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    print(f"PDF που βρέθηκαν: {len(pdf_files)}")

    for pdf_path in pdf_files:
        print(f"\n{'=' * 50}")
        print(f"Πραγματικό όνομα: {pdf_path.name}")

        reader = PdfReader(pdf_path)
        print(f"Συνολικές σελίδες PDF: {len(reader.pages)}")

        for page_index, page in enumerate(reader.pages):
            text = page.extract_text() or ""

            if text.strip():
                print(f"Πρώτη σελίδα με κείμενο: {page_index + 1}")
                print("Δείγμα κειμένου:")
                print(text[:500])
                break
        else:
            print("Δεν εντοπίστηκε κείμενο στις σελίδες.")