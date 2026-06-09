from src.db.pgvector_store import (
    fetch_item_by_id,
    hybrid_search_text_and_visual,
)
from src.engine.query_image_processor import process_query_image


def similar_search_by_item_id_pgvector(
    item_id: str,
    top_k: int = 5,
    hybrid_embeddings_path: str | None = None,
) -> list[dict]:
    """
    Find similar wardrobe items using pgvector text + visual hybrid search.

    hybrid_embeddings_path is kept only so old calls do not break.
    """
    source_item = fetch_item_by_id(item_id)

    if not source_item:
        raise ValueError(
            f"Item not found in pgvector DB: {item_id}. "
            "Run: python scripts\\migrate_json_to_pgvector.py"
        )

    results = hybrid_search_text_and_visual(
        text_embedding=source_item["text_embedding"],
        visual_embedding=source_item["visual_embedding"],
        top_k=top_k,
        exclude_item_id=item_id,
        text_weight=0.40,
        visual_weight=0.60,
    )

    return results


def similar_search_by_query_item_pgvector(
    query_item: dict,
    top_k: int = 5,
    hybrid_embeddings_path: str | None = None,
) -> list[dict]:
    """
    Find similar wardrobe items for a temporary outside image item.

    query_item comes from process_query_image() and contains:
    - text_embedding
    - visual_embedding
    """
    text_embedding = query_item.get("text_embedding")
    visual_embedding = query_item.get("visual_embedding")

    if not text_embedding:
        raise ValueError("query_item is missing text_embedding")

    if not visual_embedding:
        raise ValueError("query_item is missing visual_embedding")

    results = hybrid_search_text_and_visual(
        text_embedding=text_embedding,
        visual_embedding=visual_embedding,
        top_k=top_k,
        exclude_item_id=query_item.get("item_id"),
        text_weight=0.40,
        visual_weight=0.60,
    )

    return results


def find_similar_for_new_image_pgvector(
    image_path: str,
    top_k: int = 10,
) -> dict:
    """
    Process outside image and find similar wardrobe items using pgvector.

    This returns a simple structure similar to existing similar flow.
    """
    print("\nRunning pgvector similar search for outside image...")

    query_item = process_query_image(image_path)

    results = similar_search_by_query_item_pgvector(
        query_item=query_item,
        top_k=top_k,
    )

    return {
        "query_item": query_item,
        "recommended_items": [
            {
                "rank": index,
                "item_id": item.get("item_id"),
                "filename": item.get("filename"),
                "image_path": item.get("image_path"),
                "caption": item.get("caption"),
                "similarity_score": item.get("similarity_score"),
                "text_score": item.get("text_score"),
                "visual_score": item.get("visual_score"),
                "hybrid_pgvector_score": item.get("hybrid_pgvector_score"),
                "match_type": "similar",
                "reason": "Retrieved using pgvector hybrid text + visual similarity.",
                "similarity_explanation": (
                    "The item is similar based on both semantic caption meaning "
                    "and visual garment similarity."
                ),
                "difference_from_source": "Check exact pattern, color, fabric, and fit.",
            }
            for index, item in enumerate(results, start=1)
        ],
        "summary": (
            "Here are the closest wardrobe items found using pgvector hybrid "
            "text + visual search."
        ),
    }