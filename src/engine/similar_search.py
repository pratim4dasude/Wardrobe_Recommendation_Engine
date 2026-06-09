import json
from pathlib import Path

import torch

from src.engine.visual_embeddings import create_visual_embedding
from src.utils import resolve_project_path


HYBRID_EMBEDDINGS_PATH = "data/wardrobe_hybrid_embeddings.json"


def load_hybrid_items(
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> list[dict]:
    """
    Load wardrobe items with text and visual embeddings.
    """
    hybrid_file = resolve_project_path(hybrid_embeddings_path)

    if not hybrid_file.exists():
        raise FileNotFoundError(f"Hybrid embeddings file not found: {hybrid_file}")

    with hybrid_file.open("r", encoding="utf-8") as file:
        items = json.load(file)

    valid_items = []

    for item in items:
        if (
            item.get("status") == "hybrid_embedding_generated"
            and item.get("visual_embedding")
        ):
            valid_items.append(item)

    return valid_items


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Calculate cosine similarity between two visual embeddings.
    """
    tensor_a = torch.tensor(vector_a, dtype=torch.float32)
    tensor_b = torch.tensor(vector_b, dtype=torch.float32)

    if tensor_a.numel() != tensor_b.numel():
        raise ValueError(
            f"Vector size mismatch: {tensor_a.numel()} vs {tensor_b.numel()}"
        )

    tensor_a = tensor_a / torch.clamp(tensor_a.norm(), min=1e-12)
    tensor_b = tensor_b / torch.clamp(tensor_b.norm(), min=1e-12)

    return torch.dot(tensor_a, tensor_b).item()


def find_item_by_id(items: list[dict], item_id: str) -> dict:
    """
    Find wardrobe item by item_id.
    """
    for item in items:
        if item.get("item_id") == item_id:
            return item

    raise ValueError(f"Item ID not found: {item_id}")


def build_result_item(item: dict, score: float) -> dict:
    """
    Build clean result object.
    """
    return {
        "item_id": item.get("item_id"),
        "filename": item.get("filename"),
        "image_path": item.get("image_path"),
        "similarity_score": round(score, 4),
        "caption": item.get("caption", ""),
        "category": item.get("category", []),
        "color": item.get("color", []),
        "style": item.get("style", []),
        "search_text": item.get("search_text", ""),
    }


def similar_search_by_item_id(
    item_id: str,
    top_k: int = 5,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> list[dict]:
    """
    Scenario 1:
    Search similar wardrobe items using an existing closet item_id.
    """
    items = load_hybrid_items(hybrid_embeddings_path)

    source_item = find_item_by_id(items, item_id)
    source_embedding = source_item["visual_embedding"]

    results = []

    for item in items:
        # Do not return the same item as similar result
        if item.get("item_id") == item_id:
            continue

        score = cosine_similarity(
            source_embedding,
            item["visual_embedding"],
        )

        results.append(build_result_item(item, score))

    results = sorted(
        results,
        key=lambda item: item["similarity_score"],
        reverse=True,
    )

    return results[:top_k]


def similar_search_by_new_image(
    image_path: str | Path,
    top_k: int = 5,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> list[dict]:
    """
    Scenario 2:
    Search similar wardrobe items using a new external image.

    Example:
    User sees a dress while shopping and uploads that image.
    We create visual embedding for that image and compare with closet.
    """
    image_path = resolve_project_path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    items = load_hybrid_items(hybrid_embeddings_path)

    query_visual_embedding = create_visual_embedding(image_path)

    results = []

    for item in items:
        score = cosine_similarity(
            query_visual_embedding,
            item["visual_embedding"],
        )

        results.append(build_result_item(item, score))

    results = sorted(
        results,
        key=lambda item: item["similarity_score"],
        reverse=True,
    )

    return results[:top_k]


def print_similar_results(
    results: list[dict],
    title: str = "SIMILAR SEARCH RESULTS",
) -> None:
    """
    Print similar search results in terminal.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No similar items found.")
        return

    for rank, item in enumerate(results, start=1):
        print()
        print(f"Rank #{rank}")
        print(f"Item ID: {item['item_id']}")
        print(f"Filename: {item['filename']}")
        print(f"Image path: {item['image_path']}")
        print(f"Similarity score: {item['similarity_score']}")
        print(f"Category: {item['category']}")
        print(f"Color: {item['color']}")
        print(f"Style: {item['style']}")
        print(f"Caption: {item['caption']}")
        print("-" * 80)