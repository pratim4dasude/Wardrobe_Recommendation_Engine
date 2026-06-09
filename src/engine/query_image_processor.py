from pathlib import Path

from src.engine.captioner import caption_main_clothing, convert_caption_to_category_json
from src.engine.embeddings import create_text_embedding
from src.engine.visual_embeddings import create_visual_embedding
from src.utils import resolve_project_path


def process_query_image(image_path: str | Path) -> dict:
    """
    Process a new outside image as a temporary wardrobe-like item.

    This does NOT save the image to wardrobe_hybrid_embeddings.json.
    It only creates a temporary item for search + LLM reranking.
    """
    image_path = resolve_project_path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Query image not found: {image_path}")

    print("\nProcessing new image like a temporary wardrobe item...")
    print(f"Image path: {image_path}")

    print("\nStep 1: Generating caption...")
    raw_caption = caption_main_clothing(image_path)
    print(f"Caption: {raw_caption}")

    print("\nStep 2: Converting caption to metadata...")
    metadata = convert_caption_to_category_json(raw_caption)

    search_text = metadata.get("search_text") or metadata.get("caption") or raw_caption

    print("\nStep 3: Creating text embedding...")
    text_embedding = create_text_embedding(search_text)

    print("\nStep 4: Creating visual embedding...")
    visual_embedding = create_visual_embedding(image_path)

    query_item = {
        "item_id": "query_image",
        "filename": image_path.name,
        "image_path": str(image_path),
        "status": "temporary_query_image",

        "caption": metadata.get("caption", raw_caption),
        "category": metadata.get("category", []),
        "color": metadata.get("color", []),
        "style": metadata.get("style", []),
        "search_text": search_text,
        "raw_caption": raw_caption,

        "text_embedding_model": "text-embedding-3-small",
        "text_embedding": text_embedding,

        "visual_embedding_model": "openai/clip-vit-base-patch32",
        "visual_embedding": visual_embedding,
    }

    print("\nTemporary query item created:")
    print(f"Item ID: {query_item['item_id']}")
    print(f"Filename: {query_item['filename']}")
    print(f"Category: {query_item['category']}")
    print(f"Color: {query_item['color']}")
    print(f"Style: {query_item['style']}")
    print(f"Search text: {query_item['search_text']}")
    print(f"Text vector size: {len(query_item['text_embedding'])}")
    print(f"Visual vector size: {len(query_item['visual_embedding'])}")

    return query_item