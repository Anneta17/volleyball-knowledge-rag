# Volleyball Knowledge RAG

A beginner-friendly Retrieval-Augmented Generation (RAG) project that answers questions about volleyball using FIVB rules and coaching manuals. Answers include references to the PDF pages used.

The project runs locally with Python, FAISS, Sentence Transformers, a CrossEncoder reranker, and Ollama.

## How it works

1. Extract text from the PDF documents.
2. Split the text into chunks while keeping the PDF filename and page number.
3. Convert chunks into embeddings using `all-MiniLM-L6-v2`.
4. Store the embeddings in a FAISS index.
5. Retrieve and rerank relevant excerpts for a question.
6. Ask `qwen2.5:3b-instruct` through Ollama to generate an answer from those excerpts.
7. Display the answer and its PDF sources.

## Documents

Download these documents from FIVB and place them in `data/Pdfs/` using the filenames below. The PDFs are not included in this repository.

| Local filename | Document |
| --- | --- |
| `FIVB-Volleyball_Rules2025_2028-EN-v05.pdf` | [Official Volleyball Rules 2025–2028](https://www.fivb.com/wp-content/uploads/2025/01/FIVB-Volleyball_Rules2025_2028-EN-v05.pdf) |
| `Coaches_Manual_Level_II_EN.pdf` | [Coaches Manual Level II](https://www.fivb.com/wp-content/uploads/2024/03/Coaches_Manual_Level_II_EN.pdf) |
| `FIVB_Coach_Manual_EN.pdf` | [Coaches Manual Level I — FIVB coaching materials](https://www.fivb.com/document-category/coaches-courses-materials/) |
| `FIVB_DEV_Top_Volley_Manual_eng.pdf` | [Top Volley Manual](https://www.fivb.com/wp-content/uploads/2024/03/FIVB_DEV_Top_Volley_Manual_eng.pdf) |

Keep the local filenames exactly as listed because the scripts use them to identify sources.

## Setup on Windows

Python 3.13 was used during development. Install [Ollama for Windows](https://ollama.com/download/windows) and make sure it is running.

From PowerShell in the project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5:3b-instruct
```

Create the local directories if they do not exist:

```powershell
New-Item -ItemType Directory -Force -Path data\Pdfs
New-Item -ItemType Directory -Force -Path data\processed
```

Download the four PDFs listed above into `data\Pdfs`.

## Build the index

Run these commands in order:

```powershell
python extract_pages.py
python create_chunks.py
python create_embeddings.py
python build_index.py
```

The generated `chunks.json`, `embeddings.npy`, and `volleyball.faiss` files are stored in `data/processed/`. They are generated locally and are not committed to GitHub.

## Ask a question

```powershell
python answer.py
```

Enter a question in English. Choose `all` to search all four documents or `rules` to search only the official rules PDF.

### Example: offensive system

```text
Question: What is a 5-1 offensive system?
Sources [rules/all] (Enter = all): all

ANSWER

The 5-1 offensive system utilizes 5 hitters and 1 setter. [S1]
The 5-1 offensive system has the advantage of one player running the team offence. [S1]

SOURCES USED

[S1] Coaches_Manual_Level_II_EN.pdf — PDF page 52, chunk 3
```

### Example: official rules

```text
Question: How many liberos may be on court at the same time?
Sources [rules/all] (Enter = all): rules

ANSWER

Only one Libero may be on court at any time. [S1]

SOURCES USED

[S1] FIVB-Volleyball_Rules2025_2028-EN-v05.pdf — PDF page 44, chunk 1
```

## Known limitations

- This is an early prototype, not an authoritative interpretation of volleyball rules.
- Broad questions can produce statements that are related to a source but do not directly answer the question.
- The program checks that a supporting quote occurs in the cited chunk. This does not prove that the model interpreted the quote correctly.
- Citations currently identify PDF pages and chunks, not section titles.
- A reranker score threshold is used to reject some unrelated questions. This threshold has not been calibrated on a large evaluation set.
- Retrieval was developed and tested mainly with English questions and English documents.

## Future improvements

- Section-level metadata and citations.
- A small labeled evaluation set for answer accuracy and abstention.
- A Streamlit interface.