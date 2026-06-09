import json
import math
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

from src.engine.embeddings import create_text_embedding
from src.engine.outfit_recommender import (
    build_candidate_pool,
    detect_item_type,
    find_item_by_id,
    llm_create_outfit_recommendation,
    load_hybrid_items,
    print_outfit_recommendations,
    validate_outfit_recommendation,
)
from src.engine.query_image_processor import process_query_image
from src.engine.similar_search import similar_search_by_new_image
from src.engine.llm_similar_reranker import llm_rerank_similar_results


load_dotenv()

client = OpenAI()

FASHION_ASSISTANT_MODEL = "gpt-4.1-mini"
HYBRID_EMBEDDINGS_PATH = "data/wardrobe_hybrid_embeddings.json"


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Pure Python cosine similarity for text-only outfit search.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(f"Vector size mismatch: {len(vector_a)} vs {len(vector_b)}")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def extract_json_from_text(text: str) -> dict:
    """
    Safely parse JSON from LLM response.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        return json.loads(text[start : end + 1])

    raise ValueError("No valid JSON found in LLM response.")


def build_text_only_candidate_pool(
    user_query: str,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
    max_per_type: int = 8,
) -> dict[str, list[dict]]:
    """
    For text-only request like:
    'I need to go to a party tonight'

    Retrieve wardrobe items using semantic text embedding and group them by type.
    """
    items = load_hybrid_items(hybrid_embeddings_path)

    query_embedding = create_text_embedding(user_query)

    grouped_candidates = defaultdict(list)

    for item in items:
        item_type = detect_item_type(item)

        if item_type == "unknown":
            continue

        text_score = cosine_similarity(
            query_embedding,
            item["text_embedding"],
        )

        grouped_candidates[item_type].append(
            {
                "item_id": item.get("item_id"),
                "filename": item.get("filename"),
                "image_path": item.get("image_path"),
                "item_type": item_type,
                "candidate_score": round(text_score, 4),
                "caption": item.get("caption", ""),
                "category": item.get("category", []),
                "color": item.get("color", []),
                "style": item.get("style", []),
                "search_text": item.get("search_text", ""),
            }
        )

    final_grouped_candidates = {}

    for item_type, candidates in grouped_candidates.items():
        sorted_candidates = sorted(
            candidates,
            key=lambda item: item["candidate_score"],
            reverse=True,
        )
        final_grouped_candidates[item_type] = sorted_candidates[:max_per_type]

    return final_grouped_candidates


def llm_create_text_only_outfits(
    user_query: str,
    candidate_pool: dict[str, list[dict]],
    number_of_outfits: int = 3,
) -> dict:
    """
    Create complete outfits when user does not provide a source item.
    Example:
    'What should I wear to a party?'
    """
    prompt = f"""
You are a fashion outfit recommendation assistant.

The user request is:
"{user_query}"

Below are wardrobe items retrieved from the user's closet.
They are grouped by item type:
{json.dumps(candidate_pool, indent=2, ensure_ascii=False)}

Your task:
Create complete outfit recommendations using only the provided wardrobe items.

Very important outfit rules:
- Do not invent item IDs.
- Do not invent products outside the provided wardrobe candidates.
- Every outfit should be wearable and complete.
- If using upperwear, include bottomwear.
- If using bottomwear, include upperwear.
- Do not create an outfit with only outerwear.
- Do not create an outfit with only upperwear.
- Outerwear is optional, not mandatory.
- Use outerwear only if it improves the occasion, weather, smart-casual styling, or party styling.
- If footwear/accessory items are not available, do not force them.
- Prefer color harmony, style compatibility, occasion fit, and category compatibility.
- Give a mix of outfit types:
  1. One simple/minimal outfit.
  2. One layered or elevated outfit.
  3. One alternative style outfit.

Return only valid JSON.
Do not include markdown.

Return JSON in this exact structure:

{{
  "request_type": "text_only_occasion_outfit",
  "user_query": "{user_query}",
  "summary": "",
  "outfits": [
    {{
      "outfit_rank": 1,
      "outfit_name": "",
      "outfit_type": "minimal | layered | alternative",
      "items": [
        {{
          "role": "upperwear | bottomwear | outerwear | footwear | accessory | dress",
          "item_id": "",
          "filename": "",
          "image_path": "",
          "reason": ""
        }}
      ],
      "overall_reason": "",
      "styling_notes": "",
      "confidence": "high | medium | low"
    }}
  ]
}}

Rules:
- Return up to {number_of_outfits} outfits.
- Each outfit must include either:
  - upperwear + bottomwear, or
  - dress as the main outfit piece.
- Do not repeat the exact same outfit.
- outfit_type must be one of: minimal, layered, alternative.
- confidence must be one of: high, medium, low.
"""

    response = client.responses.create(
        model=FASHION_ASSISTANT_MODEL,
        input=prompt,
    )

    raw_text = response.output_text.strip()
    return extract_json_from_text(raw_text)


def print_text_only_outfits(recommendation: dict) -> None:
    """
    Print text-only outfit recommendations.
    """
    print("\n" + "=" * 80)
    print("TEXT-ONLY OUTFIT RECOMMENDATIONS")
    print("=" * 80)

    print(f"\nUser query: {recommendation.get('user_query')}")
    print(f"Summary: {recommendation.get('summary', '')}")

    outfits = recommendation.get("outfits", [])

    if not outfits:
        print("\nNo outfits generated.")
        return

    for outfit in outfits:
        print("\n" + "-" * 80)
        print(f"Outfit #{outfit.get('outfit_rank')}: {outfit.get('outfit_name')}")
        print(f"Outfit type: {outfit.get('outfit_type')}")
        print(f"Confidence: {outfit.get('confidence')}")
        print(f"Overall reason: {outfit.get('overall_reason')}")
        print(f"Styling notes: {outfit.get('styling_notes')}")

        print("\nItems:")
        for item in outfit.get("items", []):
            print(f"- Role: {item.get('role')}")
            print(f"  Item ID: {item.get('item_id')}")
            print(f"  Filename: {item.get('filename')}")
            print(f"  Image path: {item.get('image_path')}")
            print(f"  Reason: {item.get('reason')}")


def recommend_outfit_from_text_query(
    user_query: str,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> dict:
    """
    Case 1:
    User only gives text.
    Example:
    'I need to go to a party tonight'
    """
    print("\nBuilding wardrobe candidate pool from text query...")

    candidate_pool = build_text_only_candidate_pool(
        user_query=user_query,
        hybrid_embeddings_path=hybrid_embeddings_path,
        max_per_type=8,
    )

    for item_type, candidates in candidate_pool.items():
        print(f"{item_type}: {len(candidates)} candidate(s)")

    print("\nSending candidates to LLM outfit assistant...")

    try:
        return llm_create_text_only_outfits(
            user_query=user_query,
            candidate_pool=candidate_pool,
            number_of_outfits=3,
        )
    except Exception as error:
        return {
            "request_type": "text_only_occasion_outfit",
            "user_query": user_query,
            "summary": "Text-only outfit recommendation failed.",
            "outfits": [],
            "error": str(error),
        }


def recommend_outfit_from_wardrobe_item(
    item_id: str,
    user_query: str,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> dict:
    """
    Case 2:
    User gives wardrobe item_id and asks what to pair.
    """
    items = load_hybrid_items(hybrid_embeddings_path)
    source_item = find_item_by_id(items, item_id)

    source_type = detect_item_type(source_item)

    print("\nSource wardrobe item found:")
    print(f"Item ID: {source_item.get('item_id')}")
    print(f"Detected type: {source_type}")
    print(f"Caption: {source_item.get('caption')}")

    print("\nRetrieving compatible wardrobe items...")

    candidate_pool = build_candidate_pool(
        source_item=source_item,
        occasion=user_query,
        items=items,
        max_per_type=6,
    )

    for item_type, candidates in candidate_pool.items():
        print(f"{item_type}: {len(candidates)} candidate(s)")

    print("\nSending source item + candidates to LLM outfit recommender...")

    recommendation = llm_create_outfit_recommendation(
        source_item=source_item,
        occasion=user_query,
        candidate_pool=candidate_pool,
        number_of_outfits=3,
    )

    recommendation = validate_outfit_recommendation(
        recommendation=recommendation,
        source_item=source_item,
    )

    return recommendation


def recommend_outfit_from_new_image(
    image_path: str,
    user_query: str,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> dict:
    """
    Case 3:
    User gives outside image and asks what to pair from wardrobe.

    The outside image is processed like a temporary wardrobe item.
    """
    print("\nProcessing outside image as temporary source item...")

    query_item = process_query_image(image_path)

    items = load_hybrid_items(hybrid_embeddings_path)

    print("\nRetrieving compatible wardrobe items for this outside image...")

    candidate_pool = build_candidate_pool(
        source_item=query_item,
        occasion=user_query,
        items=items,
        max_per_type=6,
    )

    for item_type, candidates in candidate_pool.items():
        print(f"{item_type}: {len(candidates)} candidate(s)")

    print("\nSending temporary source item + candidates to LLM outfit recommender...")

    recommendation = llm_create_outfit_recommendation(
        source_item=query_item,
        occasion=user_query,
        candidate_pool=candidate_pool,
        number_of_outfits=3,
    )

    recommendation = validate_outfit_recommendation(
        recommendation=recommendation,
        source_item=query_item,
    )

    return recommendation


def find_similar_for_new_image(
    image_path: str,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> dict:
    """
    Case 4:
    User gives outside image and wants similar wardrobe items.
    """
    print("\nRunning similar search for outside image...")

    query_item, retrieved_results = similar_search_by_new_image(
        image_path=image_path,
        top_k=10,
        hybrid_embeddings_path=hybrid_embeddings_path,
    )

    print(f"\nRetrieved {len(retrieved_results)} similar candidates.")

    print("\nSending query image + candidates to LLM similar reranker...")

    reranked_response = llm_rerank_similar_results(
        source_item=query_item,
        retrieved_results=retrieved_results,
        top_k=5,
    )

    return reranked_response


def print_any_outfit_recommendation(recommendation: dict) -> None:
    """
    Print outfit recommendation depending on output format.
    """
    if recommendation.get("request_type") == "text_only_occasion_outfit":
        print_text_only_outfits(recommendation)
    else:
        print_outfit_recommendations(recommendation)