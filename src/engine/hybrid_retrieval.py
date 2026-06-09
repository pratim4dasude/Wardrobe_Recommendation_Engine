import json
from pathlib import Path

import torch
from dotenv import load_dotenv
from openai import OpenAI
from transformers import CLIPModel, CLIPProcessor

from src.utils import resolve_project_path


load_dotenv()

client = OpenAI()

HYBRID_EMBEDDINGS_PATH = "data/wardrobe_hybrid_embeddings.json"

TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

TEXT_WEIGHT = 0.70
VISUAL_WEIGHT = 0.30

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading CLIP model on: {device}")

clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)

clip_model.eval()


def load_hybrid_items(
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> list[dict]:
    """
    Load wardrobe items that contain both text and visual embeddings.
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
            and item.get("text_embedding")
            and item.get("visual_embedding")
        ):
            valid_items.append(item)

    return valid_items


def get_tensor_from_clip_output(output) -> torch.Tensor:
    """
    Some transformers versions return tensor directly.
    Some return BaseModelOutputWithPooling.

    This function safely extracts the actual tensor.
    """
    if isinstance(output, torch.Tensor):
        return output

    if hasattr(output, "pooler_output"):
        return output.pooler_output

    if hasattr(output, "last_hidden_state"):
        return output.last_hidden_state[:, 0, :]

    raise TypeError(f"Unsupported CLIP output type: {type(output)}")


def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    """
    L2 normalize tensor vector.
    """
    vector = get_tensor_from_clip_output(vector)

    norm = vector.norm(dim=-1, keepdim=True)

    # avoid divide by zero
    norm = torch.clamp(norm, min=1e-12)

    return vector / norm


def create_query_text_embedding(query: str) -> list[float]:
    """
    Create OpenAI text embedding for user query.
    This is compared with wardrobe item text_embedding.
    """
    response = client.embeddings.create(
        model=TEXT_EMBEDDING_MODEL,
        input=query,
    )

    return response.data[0].embedding


def create_query_visual_text_embedding(query: str) -> list[float]:
    """
    Create CLIP text embedding for the user query.
    This is compared with wardrobe item visual_embedding.
    """
    inputs = clip_processor(
        text=[query],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
        text_features = normalize_vector(text_features)

    return text_features[0].cpu().tolist()


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
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


def score_item(
    item: dict,
    query_text_embedding: list[float],
    query_visual_embedding: list[float],
) -> dict:
    """
    Score one wardrobe item using both:
    1. OpenAI query text embedding vs item text embedding
    2. CLIP query text embedding vs item visual embedding
    """
    text_score = cosine_similarity(
        query_text_embedding,
        item["text_embedding"],
    )

    visual_score = cosine_similarity(
        query_visual_embedding,
        item["visual_embedding"],
    )

    final_score = (TEXT_WEIGHT * text_score) + (VISUAL_WEIGHT * visual_score)

    return {
        "item_id": item["item_id"],
        "filename": item["filename"],
        "image_path": item["image_path"],
        "caption": item.get("caption", ""),
        "category": item.get("category", []),
        "color": item.get("color", []),
        "style": item.get("style", []),
        "search_text": item.get("search_text", ""),
        "text_score": round(text_score, 4),
        "visual_score": round(visual_score, 4),
        "final_score": round(final_score, 4),
    }


def hybrid_search(
    query: str,
    top_k: int = 5,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> list[dict]:
    """
    Run hybrid retrieval using:
    - OpenAI text embedding
    - CLIP visual-text embedding
    """
    items = load_hybrid_items(hybrid_embeddings_path)

    if not items:
        raise ValueError("No valid hybrid items found.")

    print(f"\nQuery: {query}")
    print(f"Loaded {len(items)} hybrid wardrobe items.")

    print("Creating OpenAI text query embedding...")
    query_text_embedding = create_query_text_embedding(query)

    print("Creating CLIP visual-text query embedding...")
    query_visual_embedding = create_query_visual_text_embedding(query)

    print("Scoring wardrobe items...")

    scored_items = []

    for item in items:
        try:
            scored_item = score_item(
                item=item,
                query_text_embedding=query_text_embedding,
                query_visual_embedding=query_visual_embedding,
            )
            scored_items.append(scored_item)

        except Exception as error:
            print(f"Skipping {item.get('item_id')}: {error}")

    scored_items = sorted(
        scored_items,
        key=lambda item: item["final_score"],
        reverse=True,
    )

    return scored_items[:top_k]


def print_search_results(results: list[dict]) -> None:
    """
    Print search results clearly in terminal.
    """
    print("\n" + "=" * 80)
    print("HYBRID SEARCH RESULTS")
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for rank, item in enumerate(results, start=1):
        print()
        print(f"Rank #{rank}")
        print(f"Item ID: {item['item_id']}")
        print(f"Filename: {item['filename']}")
        print(f"Image path: {item['image_path']}")
        print(f"Final score: {item['final_score']}")
        print(f"Text score: {item['text_score']}")
        print(f"Visual score: {item['visual_score']}")
        print(f"Category: {item['category']}")
        print(f"Color: {item['color']}")
        print(f"Style: {item['style']}")
        print(f"Caption: {item['caption']}")
        print(f"Search text: {item['search_text']}")
        print("-" * 80)