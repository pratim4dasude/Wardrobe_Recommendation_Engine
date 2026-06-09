import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.utils import resolve_project_path


load_dotenv()

client = OpenAI()

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_SLEEP_SECONDS = 0.5


def create_text_embedding(text: str) -> list[float]:
    """
    Create OpenAI text embedding from search_text.
    """
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )

    return response.data[0].embedding


def build_embedding_item(item: dict) -> dict:
    """
    Convert one wardrobe metadata item into embedding-ready item.
    """
    search_text = item.get("search_text") or item.get("caption") or ""

    if not search_text:
        raise ValueError(f"No search_text found for {item.get('item_id')}")

    embedding = create_text_embedding(search_text)

    return {
        "item_id": item["item_id"],
        "filename": item["filename"],
        "image_path": item["image_path"],
        "caption": item.get("caption", ""),
        "category": item.get("category", []),
        "color": item.get("color", []),
        "style": item.get("style", []),
        "search_text": search_text,
        "embedding_model": EMBEDDING_MODEL,
        "text_embedding": embedding,
    }


def save_embeddings(output_file: Path, results: list[dict]) -> None:
    """
    Save embeddings after every item so progress is not lost.
    """
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def generate_embeddings_from_metadata(
    metadata_path: str = "data/wardrobe_metadata.json",
    output_path: str = "data/wardrobe_embeddings.json",
) -> None:
    """
    Read wardrobe metadata and generate text embeddings from search_text.
    """
    metadata_file = resolve_project_path(metadata_path)
    output_file = resolve_project_path(output_path)

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with metadata_file.open("r", encoding="utf-8") as file:
        metadata_items = json.load(file)

    results = []

    for index, item in enumerate(metadata_items, start=1):
        print("\n" + "=" * 80)
        print(f"[{index}/{len(metadata_items)}] Creating text embedding for {item['filename']}")
        print("=" * 80)

        try:
            print(f"Item ID: {item['item_id']}")
            print(f"Search text: {item.get('search_text', '')}")

            embedded_item = build_embedding_item(item)
            results.append(embedded_item)

            save_embeddings(output_file, results)

            print("\nEmbedding created successfully.")
            print(f"Embedding model: {EMBEDDING_MODEL}")
            print(f"Vector size: {len(embedded_item['text_embedding'])}")
            print(f"Progress: {index}/{len(metadata_items)} completed")

        except Exception as error:
            print(f"\nFailed to create embedding for {item.get('item_id')}: {error}")

            failed_item = {
                "item_id": item.get("item_id", "unknown"),
                "filename": item.get("filename", "unknown"),
                "image_path": item.get("image_path", ""),
                "status": "embedding_failed",
                "error": str(error),
            }

            results.append(failed_item)
            save_embeddings(output_file, results)

        time.sleep(EMBEDDING_SLEEP_SECONDS)

    print("\n" + "=" * 80)
    print(f"Text embeddings generated for {len(results)} item(s).")
    print(f"Saved at: {output_file}")
    print("=" * 80)