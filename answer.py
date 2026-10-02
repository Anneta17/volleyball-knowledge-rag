import json
from collections import defaultdict
from pathlib import Path

import faiss
from ollama import chat
from sentence_transformers import CrossEncoder, SentenceTransformer


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RULES_PDF = "FIVB-Volleyball_Rules2025_2028-EN-v05.pdf"
OLLAMA_MODEL = "qwen2.5:3b-instruct"

MAX_PAGES = 5
MAX_CONTEXT_CHUNKS = 5


def normalize_text(text):
    """Ignore differences in whitespace and capitalization."""
    return " ".join(text.split()).casefold()


def load_data():
    chunks_path = PROCESSED_DIR / "chunks.json"
    index_path = PROCESSED_DIR / "volleyball.faiss"

    if not chunks_path.exists() or not index_path.exists():
        raise SystemExit(
            "Missing chunks.json or volleyball.faiss. "
            "Run the data preparation scripts first."
        )

    with chunks_path.open(encoding="utf-8") as file:
        chunks = json.load(file)

    index = faiss.read_index(str(index_path))

    if index.ntotal != len(chunks):
        raise SystemExit(
            "The number of chunks does not match the FAISS index. "
            "Rebuild the embeddings and index."
        )

    return chunks, index


def retrieve(question, mode, chunks, index):
    """Use FAISS to identify relevant PDF pages."""
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    query_vector = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    scores, positions = index.search(query_vector, index.ntotal)

    chunks_by_page = defaultdict(list)

    for chunk in chunks:
        page_key = (chunk["source"], chunk["page"])
        chunks_by_page[page_key].append(chunk)

    selected_pages = []
    seen_pages = set()

    for position in positions[0]:
        if position < 0:
            continue

        chunk = chunks[int(position)]

        if chunk["text"].count(".....") >= 2:
            continue

        if mode == "rules" and chunk["source"] != RULES_PDF:
            continue

        page_key = (chunk["source"], chunk["page"])

        if page_key in seen_pages:
            continue

        seen_pages.add(page_key)
        selected_pages.append(page_key)

        if len(selected_pages) == MAX_PAGES:
            break

    candidates = []

    for page_key in selected_pages:
        for chunk in chunks_by_page[page_key]:
            if chunk["text"].count(".....") < 2:
                candidates.append(chunk)

    return candidates


def rerank(question, candidates):
    """Rerank the candidate chunks with a CrossEncoder."""
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")

    pairs = [
        (question, chunk["text"])
        for chunk in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda pair: float(pair[0]),
        reverse=True,
    )

    results = []

    for score, chunk in ranked[:MAX_CONTEXT_CHUNKS]:
        results.append({
            "score": float(score),
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_number": chunk["chunk_number"],
            "text": chunk["text"],
        })

    return results


def generate_claims(question, results):
    """Ask Ollama for statements with source IDs and supporting quotes."""
    source_ids = [
        f"S{number}"
        for number in range(1, len(results) + 1)
    ]

    schema = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "maxItems": 3,
                "items": {
                    "type": "object",
                    "properties": {
                        "statement": {"type": "string"},
                        "source_id": {
                            "type": "string",
                            "enum": source_ids,
                        },
                        "quote": {"type": "string"},
                    },
                    "required": [
                        "statement",
                        "source_id",
                        "quote",
                    ],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["claims"],
        "additionalProperties": False,
    }

    context = "\n\n".join(
        f"[S{number}] {item['source']}, "
        f"PDF page {item['page']}, "
        f"chunk {item['chunk_number']}\n"
        f"{item['text']}"
        for number, item in enumerate(results, start=1)
    )

    response = chat(
        model=OLLAMA_MODEL,
        format=schema,
        options={"temperature": 0},
        messages=[
            {
                "role": "system",
                "content": (
                    "Always write the statements in English, even if the "
                    "question is written in another language. "
                    "Answer the question using only the excerpts. "
                    "Return at most three short factual statements. "
                    "For each statement, provide its exact source_id and "
                    "a short verbatim quote copied from that source which "
                    "directly supports the statement. "
                    "Do not put source IDs inside statements. "
                    "Every statement must directly answer the exact question. "
                    "If asked what someone can do, describe only actions that "
                    "person performs, not actions others perform about that person. "
                    "Ignore related facts that do not answer the question. "
                    "If the excerpts do not answer the question, "
                    "return an empty claims list."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question}\n\n"
                    f"Excerpts:\n{context}"
                ),
            },
        ],
    )

    try:
        data = json.loads(response.message.content or "{}")
    except json.JSONDecodeError:
        return []

    claims = data.get("claims", [])

    if not isinstance(claims, list):
        return []

    return claims


def validate_claims(claims, results):
    """Keep claims whose source ID and supporting quote are valid."""
    sources = {
        f"S{number}": item
        for number, item in enumerate(results, start=1)
    }

    valid_claims = []

    for claim in claims:
        if not isinstance(claim, dict):
            continue

        statement = claim.get("statement")
        source_id = claim.get("source_id")
        quote = claim.get("quote")

        if not all(
            isinstance(value, str)
            for value in (statement, source_id, quote)
        ):
            continue

        statement = statement.strip()
        quote = quote.strip()
        source = sources.get(source_id)

        if not statement or source is None:
            continue

        if len(normalize_text(quote)) < 20:
            continue

        if normalize_text(quote) not in normalize_text(source["text"]):
            continue

        valid_claims.append({
            "statement": statement,
            "source_id": source_id,
            "quote": quote,
        })

    return valid_claims


def main():
    question = input("Question: ").strip()

    if not question:
        raise SystemExit("Please enter a question.")

    mode = (
        input("Sources [rules/all] (Enter = all): ")
        .strip()
        .lower()
        or "all"
    )

    if mode not in {"rules", "all"}:
        raise SystemExit("Please choose 'rules' or 'all'.")

    chunks, index = load_data()
    candidates = retrieve(question, mode, chunks, index)

    if not candidates:
        raise SystemExit("No candidate excerpts were found.")

    results = rerank(question, candidates)

    if results[0]["score"] < -5.0:
        print("\nANSWER\n")
        print(
            "I could not find enough information in the provided "
            "documents to answer this question."
        )
        return

    print("\nCHUNKS PROVIDED TO THE MODEL\n")

    for number, item in enumerate(results, start=1):
        print(
            f"[S{number}] {item['source']} — "
            f"PDF page {item['page']}, "
            f"chunk {item['chunk_number']}, "
            f"rerank score {item['score']:.3f}"
        )

    claims = generate_claims(question, results)
    valid_claims = validate_claims(claims, results)

    print("\nANSWER\n")

    if not valid_claims:
        print(
            "I could not produce an answer with verifiable "
            "citations from the retrieved excerpts."
        )
        return

    for claim in valid_claims:
        statement = claim["statement"]

        if statement[-1] not in ".!?":
            statement += "."

        print(f"{statement} [{claim['source_id']}]")

    print("\nSOURCES USED\n")

    used_ids = {
        claim["source_id"]
        for claim in valid_claims
    }

    for number, item in enumerate(results, start=1):
        source_id = f"S{number}"

        if source_id in used_ids:
            print(
                f"[{source_id}] {item['source']} — "
                f"PDF page {item['page']}, "
                f"chunk {item['chunk_number']}"
            )


if __name__ == "__main__":
    main()