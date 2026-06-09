import json
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from src.utils import resolve_project_path


CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
VISUAL_SLEEP_SECONDS = 0.3

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP model on: {device}")

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

clip_model.eval()


def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    """
    L2 normalize tensor vector.
    This makes cosine similarity easier later.
    """
    return vector / vector.norm(dim=-1, keepdim=True)


def create_visual_embedding(image_path: str | Path) -> list[float]:
    """
    Create visual embedding from wardrobe image using CLIP image encoder.
    """
    image_path = resolve_project_path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")

    inputs = clip_processor(
        images=image,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

        # Safety check:
        # Some model outputs may return an object instead of direct tensor.
        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output

        image_features = normalize_vector(image_features)

    return image_features[0].cpu().tolist()


def build_hybrid_item(item: dict) -> dict:
    """
    Add visual embedding to existing text embedding item.
    """
    visual_embedding = create_visual_embedding(item["image_path"])

    return {
        "item_id": item["item_id"],
        "filename": item["filename"],
        "image_path": item["image_path"],

        "caption": item.get("caption", ""),
        "category": item.get("category", []),
        "color": item.get("color", []),
        "style": item.get("style", []),
        "search_text": item.get("search_text", ""),

        "text_embedding_model": item.get("embedding_model", ""),
        "text_embedding": item.get("text_embedding", []),

        "visual_embedding_model": CLIP_MODEL_NAME,
        "visual_embedding": visual_embedding,

        "status": "hybrid_embedding_generated",
    }


def save_hybrid_embeddings(output_file: Path, results: list[dict]) -> None:
    """
    Save hybrid embeddings after every item.
    """
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def generate_visual_and_hybrid_embeddings(
    text_embeddings_path: str = "data/wardrobe_embeddings.json",
    output_path: str = "data/wardrobe_hybrid_embeddings.json",
) -> None:
    """
    Reads existing text embeddings and adds visual embeddings.
    Final output contains both text_embedding and visual_embedding.
    """
    text_embeddings_file = resolve_project_path(text_embeddings_path)
    output_file = resolve_project_path(output_path)

    if not text_embeddings_file.exists():
        raise FileNotFoundError(
            f"Text embeddings file not found: {text_embeddings_file}"
        )

    with text_embeddings_file.open("r", encoding="utf-8") as file:
        text_embedding_items = json.load(file)

    results = []

    for index, item in enumerate(text_embedding_items, start=1):
        print("\n" + "=" * 80)
        print(f"[{index}/{len(text_embedding_items)}] Creating visual embedding for {item['filename']}")
        print("=" * 80)

        try:
            hybrid_item = build_hybrid_item(item)
            results.append(hybrid_item)

            save_hybrid_embeddings(output_file, results)

            print(f"Item ID: {item['item_id']}")
            print(f"Text vector size: {len(hybrid_item['text_embedding'])}")
            print(f"Visual vector size: {len(hybrid_item['visual_embedding'])}")
            print(f"Status: {hybrid_item['status']}")
            print(f"Progress: {index}/{len(text_embedding_items)} completed")
            print("Hybrid item saved successfully.")

        except Exception as error:
            print(f"Failed to create visual embedding for {item.get('item_id')}: {error}")

            failed_item = {
                "item_id": item.get("item_id", "unknown"),
                "filename": item.get("filename", "unknown"),
                "image_path": item.get("image_path", ""),
                "status": "visual_embedding_failed",
                "caption": item.get("caption", ""),
                "category": item.get("category", []),
                "color": item.get("color", []),
                "style": item.get("style", []),
                "search_text": item.get("search_text", ""),
                "text_embedding_model": item.get("embedding_model", ""),
                "text_embedding": item.get("text_embedding", []),
                "visual_embedding_model": CLIP_MODEL_NAME,
                "visual_embedding": [],
                "error": str(error),
            }

            results.append(failed_item)
            save_hybrid_embeddings(output_file, results)

        time.sleep(VISUAL_SLEEP_SECONDS)

    print("\n" + "=" * 80)
    print(f"Hybrid embeddings generated for {len(results)} item(s).")
    print(f"Saved at: {output_file}")
    print("=" * 80)