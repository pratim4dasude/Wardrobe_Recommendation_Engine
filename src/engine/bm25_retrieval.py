import math
import re
from collections import Counter


def tokenize_text(text: str) -> list[str]:
    """
    Simple tokenizer for wardrobe search text.
    """
    if not text:
        return []

    text = text.lower()
    return re.findall(r"[a-z0-9]+", text)


def build_item_document(item: dict) -> str:
    """
    Convert one wardrobe item into searchable text for BM25.
    """
    parts = []

    for key in ["caption", "search_text", "filename"]:
        value = item.get(key)
        if isinstance(value, str):
            parts.append(value)

    for key in ["category", "color", "style", "occasion"]:
        value = item.get(key)

        if isinstance(value, list):
            parts.extend(str(v) for v in value)
        elif isinstance(value, str):
            parts.append(value)

    return " ".join(parts)


def compute_bm25_scores(
    query: str,
    items: list[dict],
    k1: float = 1.5,
    b: float = 0.75,
) -> list[float]:
    """
    Compute BM25 scores for a list of wardrobe items.
    No external library required.
    """
    query_tokens = tokenize_text(query)

    if not query_tokens or not items:
        return [0.0 for _ in items]

    documents = [tokenize_text(build_item_document(item)) for item in items]
    document_lengths = [len(doc) for doc in documents]

    average_document_length = (
        sum(document_lengths) / len(document_lengths)
        if document_lengths
        else 0.0
    )

    if average_document_length == 0:
        return [0.0 for _ in items]

    document_frequencies = Counter()

    for doc in documents:
        for token in set(doc):
            document_frequencies[token] += 1

    total_documents = len(documents)
    scores = []

    for doc, doc_length in zip(documents, document_lengths, strict=False):
        token_frequencies = Counter(doc)
        score = 0.0

        for token in query_tokens:
            if token not in token_frequencies:
                continue

            document_frequency = document_frequencies.get(token, 0)

            idf = math.log(
                1
                + (
                    total_documents - document_frequency + 0.5
                )
                / (document_frequency + 0.5)
            )

            term_frequency = token_frequencies[token]

            denominator = term_frequency + k1 * (
                1 - b + b * (doc_length / average_document_length)
            )

            score += idf * (
                (term_frequency * (k1 + 1)) / denominator
            )

        scores.append(round(score, 4))

    return scores


def normalize_scores(scores: list[float]) -> list[float]:
    """
    Normalize scores between 0 and 1.
    """
    if not scores:
        return []

    max_score = max(scores)

    if max_score == 0:
        return [0.0 for _ in scores]

    return [round(score / max_score, 4) for score in scores]


def get_existing_score(item: dict) -> float:
    """
    Read existing score from different retrieval flows.
    """
    for key in [
        "similarity_score",
        "score",
        "metadata_score",
        "compatibility_score",
        "final_score",
        "hybrid_score",
    ]:
        value = item.get(key)

        if isinstance(value, (int, float)):
            return float(value)

    return 0.0


def rerank_items_with_bm25(
    query: str,
    items: list[dict],
    bm25_weight: float = 0.30,
    existing_score_weight: float = 0.70,
) -> list[dict]:
    """
    Add BM25 score and hybrid keyword score to candidates.

    hybrid_keyword_score = existing score + BM25 exact keyword score
    """
    if not items:
        return []

    bm25_raw_scores = compute_bm25_scores(query=query, items=items)
    bm25_scores = normalize_scores(bm25_raw_scores)

    reranked_items = []

    for item, bm25_raw, bm25_score in zip(
        items,
        bm25_raw_scores,
        bm25_scores,
        strict=False,
    ):
        item_copy = dict(item)

        existing_score = get_existing_score(item_copy)

        hybrid_score = (
            existing_score_weight * existing_score
            + bm25_weight * bm25_score
        )

        item_copy["bm25_raw_score"] = bm25_raw
        item_copy["bm25_score"] = bm25_score
        item_copy["hybrid_keyword_score"] = round(hybrid_score, 4)

        reranked_items.append(item_copy)

    reranked_items.sort(
        key=lambda item: item.get("hybrid_keyword_score", 0.0),
        reverse=True,
    )

    return reranked_items