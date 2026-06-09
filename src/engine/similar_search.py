import json
from pathlib import Path

import torch

from src.engine.query_image_processor import process_query_image
from src.utils import resolve_project_path


HYBRID_EMBEDDINGS_PATH = "data/wardrobe_hybrid_embeddings.json"

# For new uploaded image, use both text and visual signal.
TEXT_WEIGHT_FOR_NEW_IMAGE = 0.40
VISUAL_WEIGHT_FOR_NEW_IMAGE = 0.60


def load_hybrid_items(
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> list[dict]:
    """
    Load wardrobe items with both text and visual embeddings.
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
            and item.get("text_embedding")
        ):
            valid_items.append(item)

    return valid_items


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Calculate cosine similarity between two embeddings.
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


def build_result_item(
    item: dict,
    similarity_score: float,
    text_score: float | None = None,
    visual_score: float | None = None,
) -> dict:
    """
    Build clean result object for similar search output.
    """
    result = {
        "item_id": item.get("item_id"),
        "filename": item.get("filename"),
        "image_path": item.get("image_path"),
        "similarity_score": round(similarity_score, 4),
        "caption": item.get("caption", ""),
        "category": item.get("category", []),
        "color": item.get("color", []),
        "style": item.get("style", []),
        "search_text": item.get("search_text", ""),
    }

    if text_score is not None:
        result["text_score"] = round(text_score, 4)

    if visual_score is not None:
        result["visual_score"] = round(visual_score, 4)

    return result


def similar_search_by_item_id(
    item_id: str,
    top_k: int = 5,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> list[dict]:
    """
    Option 1:
    Existing wardrobe item_id.

    This uses saved visual_embedding from wardrobe_hybrid_embeddings.json.
    """
    items = load_hybrid_items(hybrid_embeddings_path)

    source_item = find_item_by_id(items, item_id)
    source_visual_embedding = source_item["visual_embedding"]

    results = []

    for item in items:
        if item.get("item_id") == item_id:
            continue

        visual_score = cosine_similarity(
            source_visual_embedding,
            item["visual_embedding"],
        )

        results.append(
            build_result_item(
                item=item,
                similarity_score=visual_score,
                visual_score=visual_score,
            )
        )

    results = sorted(
        results,
        key=lambda item: item["similarity_score"],
        reverse=True,
    )

    return results[:top_k]


def similar_search_by_query_item(
    query_item: dict,
    top_k: int = 5,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> list[dict]:
    """
    Search wardrobe using a temporary query item.

    This is used for new uploaded images after we create:
    - caption
    - metadata
    - text_embedding
    - visual_embedding
    """
    items = load_hybrid_items(hybrid_embeddings_path)

    results = []

    for item in items:
        text_score = cosine_similarity(
            query_item["text_embedding"],
            item["text_embedding"],
        )

        visual_score = cosine_similarity(
            query_item["visual_embedding"],
            item["visual_embedding"],
        )

        final_score = (
            TEXT_WEIGHT_FOR_NEW_IMAGE * text_score
            + VISUAL_WEIGHT_FOR_NEW_IMAGE * visual_score
        )

        results.append(
            build_result_item(
                item=item,
                similarity_score=final_score,
                text_score=text_score,
                visual_score=visual_score,
            )
        )

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
) -> tuple[dict, list[dict]]:
    """
    Option 2:
    New outside image.

    We first make the new image look like a temporary wardrobe item.
    Then we search using both text + visual embeddings.
    """
    image_path = resolve_project_path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    query_item = process_query_image(image_path)

    results = similar_search_by_query_item(
        query_item=query_item,
        top_k=top_k,
        hybrid_embeddings_path=hybrid_embeddings_path,
    )

    return query_item, results


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

        if "text_score" in item:
            print(f"Text score: {item['text_score']}")

        if "visual_score" in item:
            print(f"Visual score: {item['visual_score']}")

        print(f"Category: {item['category']}")
        print(f"Color: {item['color']}")
        print(f"Style: {item['style']}")
        print(f"Caption: {item['caption']}")
        print("-" * 80)