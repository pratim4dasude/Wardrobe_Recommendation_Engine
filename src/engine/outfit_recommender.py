import json
import math
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

from src.engine.embeddings import create_text_embedding
from src.utils import resolve_project_path


load_dotenv()

client = OpenAI()

OUTFIT_MODEL = "gpt-4.1-mini"
HYBRID_EMBEDDINGS_PATH = "data/wardrobe_hybrid_embeddings.json"


def load_hybrid_items(
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> list[dict]:
    """
    Load wardrobe items with metadata and embeddings.
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
            and item.get("caption")
        ):
            valid_items.append(item)

    return valid_items


def find_item_by_id(items: list[dict], item_id: str) -> dict:
    """
    Find source wardrobe item.
    """
    for item in items:
        if item.get("item_id") == item_id:
            return item

    raise ValueError(f"Item ID not found: {item_id}")


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """
    Pure Python cosine similarity.
    """
    if len(vector_a) != len(vector_b):
        raise ValueError(f"Vector size mismatch: {len(vector_a)} vs {len(vector_b)}")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def detect_item_type(item: dict) -> str:
    """
    Detect broad fashion type from category metadata, caption, and search text.
    """
    category_text = " ".join(item.get("category", [])).lower()
    caption_text = item.get("caption", "").lower()
    search_text = item.get("search_text", "").lower()

    full_text = f"{category_text} {caption_text} {search_text}"

    if any(
        word in full_text
        for word in ["jeans", "trousers", "pants", "shorts", "bottomwear", "bottom wear"]
    ):
        return "bottomwear"

    if any(
        word in full_text
        for word in ["sneakers", "shoes", "footwear", "sandals", "boots", "loafers"]
    ):
        return "footwear"

    if any(
        word in full_text
        for word in ["jacket", "blazer", "coat", "outerwear", "hoodie", "sweater"]
    ):
        return "outerwear"

    if any(word in full_text for word in ["dress", "gown"]):
        return "dress"

    if any(
        word in full_text
        for word in ["accessory", "bag", "belt", "watch", "cap", "hat"]
    ):
        return "accessory"

    if any(
        word in full_text
        for word in [
            "shirt",
            "t-shirt",
            "tee",
            "tank top",
            "polo",
            "top",
            "upperwear",
            "upper wear",
            "sleeveless",
        ]
    ):
        return "upperwear"

    return "unknown"


def get_target_item_types(source_type: str) -> list[str]:
    """
    Decide what item types are useful to complete an outfit.
    These are candidate types, not mandatory types.
    """
    compatibility_map = {
        "upperwear": ["bottomwear", "footwear", "outerwear", "accessory"],
        "bottomwear": ["upperwear", "footwear", "outerwear", "accessory"],
        "dress": ["footwear", "outerwear", "accessory"],
        "footwear": ["upperwear", "bottomwear", "dress", "outerwear"],
        "outerwear": ["upperwear", "bottomwear", "footwear", "accessory"],
        "accessory": ["upperwear", "bottomwear", "dress", "footwear"],
        "unknown": ["upperwear", "bottomwear", "footwear", "outerwear", "accessory"],
    }

    return compatibility_map.get(source_type, compatibility_map["unknown"])


def get_required_roles_for_source_type(source_type: str) -> list[str]:
    """
    Decide mandatory roles for a complete outfit.

    Example:
    - If source is upperwear, bottomwear is required.
    - If source is bottomwear, upperwear is required.
    - If source is dress, no upper/lower required because dress is already a main outfit piece.
    """
    required_roles_map = {
        "upperwear": ["bottomwear"],
        "bottomwear": ["upperwear"],
        "dress": [],
        "footwear": ["upperwear", "bottomwear"],
        "outerwear": ["upperwear", "bottomwear"],
        "accessory": ["upperwear", "bottomwear"],
        "unknown": ["upperwear", "bottomwear"],
    }

    return required_roles_map.get(source_type, required_roles_map["unknown"])


def metadata_bonus_score(source_item: dict, candidate_item: dict, occasion: str) -> float:
    """
    Small rule-based score to help candidate selection before LLM.
    LLM still makes the final decision.
    """
    score = 0.0

    source_styles = set(style.lower() for style in source_item.get("style", []))
    candidate_styles = set(style.lower() for style in candidate_item.get("style", []))

    style_overlap = source_styles.intersection(candidate_styles)
    score += len(style_overlap) * 0.05

    occasion_text = occasion.lower()
    candidate_text = (
        " ".join(candidate_item.get("style", []))
        + " "
        + candidate_item.get("caption", "")
        + " "
        + candidate_item.get("search_text", "")
    ).lower()

    for word in occasion_text.split():
        if word in candidate_text:
            score += 0.03

    return score


def build_candidate_pool(
    source_item: dict,
    occasion: str,
    items: list[dict],
    max_per_type: int = 6,
) -> dict[str, list[dict]]:
    """
    Retrieve compatible candidates grouped by outfit role/type.
    Uses text embedding similarity + lightweight metadata score.
    """
    source_type = detect_item_type(source_item)
    target_types = get_target_item_types(source_type)

    outfit_query = (
        f"Outfit item compatible with: {source_item.get('caption', '')}. "
        f"Occasion/style intent: {occasion}. "
        f"Source colors: {source_item.get('color', [])}. "
        f"Source style: {source_item.get('style', [])}."
    )

    query_embedding = create_text_embedding(outfit_query)

    grouped_candidates = defaultdict(list)

    for item in items:
        if item.get("item_id") == source_item.get("item_id"):
            continue

        candidate_type = detect_item_type(item)

        if candidate_type not in target_types:
            continue

        text_score = cosine_similarity(query_embedding, item["text_embedding"])
        bonus_score = metadata_bonus_score(source_item, item, occasion)
        final_score = text_score + bonus_score

        grouped_candidates[candidate_type].append(
            {
                "item_id": item.get("item_id"),
                "filename": item.get("filename"),
                "image_path": item.get("image_path"),
                "item_type": candidate_type,
                "candidate_score": round(final_score, 4),
                "text_score": round(text_score, 4),
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


def build_source_context(source_item: dict) -> dict:
    """
    Prepare source item for LLM.
    """
    return {
        "item_id": source_item.get("item_id"),
        "filename": source_item.get("filename"),
        "image_path": source_item.get("image_path"),
        "item_type": detect_item_type(source_item),
        "caption": source_item.get("caption", ""),
        "category": source_item.get("category", []),
        "color": source_item.get("color", []),
        "style": source_item.get("style", []),
        "search_text": source_item.get("search_text", ""),
    }


def extract_json_from_text(text: str) -> dict:
    """
    Parse JSON safely from LLM response.
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


def llm_create_outfit_recommendation(
    source_item: dict,
    occasion: str,
    candidate_pool: dict[str, list[dict]],
    number_of_outfits: int = 3,
) -> dict:
    """
    Ask LLM to create complete outfit recommendations from retrieved candidates.
    """
    source_context = build_source_context(source_item)

    source_type = detect_item_type(source_item)
    required_roles = get_required_roles_for_source_type(source_type)

    prompt = f"""
You are a fashion outfit recommendation assistant.

The user selected this source wardrobe item:
{json.dumps(source_context, indent=2, ensure_ascii=False)}

Detected source item type:
"{source_type}"

Required roles for a complete outfit:
{json.dumps(required_roles, indent=2, ensure_ascii=False)}

The user wants an outfit for this occasion/style intent:
"{occasion}"

Below are compatible wardrobe candidates retrieved from the user's closet.
They are grouped by item type:
{json.dumps(candidate_pool, indent=2, ensure_ascii=False)}

Your task:
Create practical and complete outfit recommendations using only the provided wardrobe items.

Very important outfit rules:
- Do not invent item IDs.
- Do not invent products outside the provided wardrobe candidates.
- The selected source item must be included in every outfit.
- If the source item is upperwear, every outfit must include at least one bottomwear item.
- If the source item is bottomwear, every outfit must include at least one upperwear item.
- If the source item is outerwear, every outfit must include upperwear and bottomwear if available.
- If the source item is footwear or accessory, every outfit must include upperwear and bottomwear if available.
- Do not create an outfit with only source + outerwear.
- Do not create an outfit with only upperwear items.
- Do not create an outfit with only outerwear items.
- Outerwear is optional, not mandatory, unless the source item itself is outerwear.
- Use outerwear only when it improves the occasion, weather, smart-casual styling, or formal styling.
- Prefer complete wearable outfits over over-layered outfits.
- If footwear/accessory items are not available, do not force them.
- Prefer color harmony, style compatibility, occasion fit, and category compatibility.

Give a mix of outfit types:
1. One minimal outfit: source + required lower/upper piece.
2. One layered outfit: source + required piece + optional outerwear, only if it improves the outfit.
3. One alternative outfit: different color/style combination if available.

Return only valid JSON.
Do not include markdown.

Return JSON in this exact structure:

{{
  "source_item_id": "{source_item.get('item_id')}",
  "source_item_type": "{source_type}",
  "occasion": "{occasion}",
  "summary": "",
  "outfits": [
    {{
      "outfit_rank": 1,
      "outfit_name": "",
      "outfit_type": "minimal | layered | alternative",
      "items": [
        {{
          "role": "source | upperwear | bottomwear | outerwear | footwear | accessory",
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
- Each outfit should include the source item and at least the required role items.
- For upperwear source, do not return any outfit without bottomwear.
- For bottomwear source, do not return any outfit without upperwear.
- Outerwear/blazer/jacket should be treated as an optional layer.
- Do not repeat the exact same outfit.
- outfit_type must be one of: minimal, layered, alternative.
- confidence must be one of: high, medium, low.
"""

    response = client.responses.create(
        model=OUTFIT_MODEL,
        input=prompt,
    )

    raw_text = response.output_text.strip()

    return extract_json_from_text(raw_text)


def validate_outfit_recommendation(recommendation: dict, source_item: dict) -> dict:
    """
    Lightweight post-validation to remove incomplete outfits.

    This keeps the output safe even if the LLM ignores a rule.
    """
    source_type = detect_item_type(source_item)
    required_roles = get_required_roles_for_source_type(source_type)

    valid_outfits = []

    for outfit in recommendation.get("outfits", []):
        items = outfit.get("items", [])
        roles = [item.get("role", "").lower() for item in items]

        has_source = any(
            item.get("item_id") == source_item.get("item_id")
            for item in items
        )

        if not has_source:
            continue

        missing_required_role = False

        for required_role in required_roles:
            if required_role not in roles:
                missing_required_role = True
                break

        if missing_required_role:
            continue

        # Avoid bad outfit like shirt + blazer only.
        if source_type == "upperwear" and "bottomwear" not in roles:
            continue

        if source_type == "bottomwear" and "upperwear" not in roles:
            continue

        valid_outfits.append(outfit)

    recommendation["outfits"] = valid_outfits

    if not valid_outfits:
        recommendation["summary"] = (
            recommendation.get("summary", "")
            + " No complete outfits passed validation."
        )

    return recommendation


def recommend_outfits_for_item(
    item_id: str,
    occasion: str,
    hybrid_embeddings_path: str = HYBRID_EMBEDDINGS_PATH,
) -> dict:
    """
    Main function:
    item_id + occasion -> outfit recommendations.
    """
    items = load_hybrid_items(hybrid_embeddings_path)
    source_item = find_item_by_id(items, item_id)

    source_type = detect_item_type(source_item)
    target_types = get_target_item_types(source_type)
    required_roles = get_required_roles_for_source_type(source_type)

    print("\nSource item found:")
    print(f"Item ID: {source_item.get('item_id')}")
    print(f"Detected type: {source_type}")
    print(f"Target compatible types: {target_types}")
    print(f"Required roles for complete outfit: {required_roles}")
    print(f"Caption: {source_item.get('caption')}")

    print("\nRetrieving compatible outfit candidates...")

    candidate_pool = build_candidate_pool(
        source_item=source_item,
        occasion=occasion,
        items=items,
        max_per_type=6,
    )

    for item_type, candidates in candidate_pool.items():
        print(f"{item_type}: {len(candidates)} candidate(s)")

    print("\nSending source item + candidates to LLM outfit recommender...")

    try:
        recommendation = llm_create_outfit_recommendation(
            source_item=source_item,
            occasion=occasion,
            candidate_pool=candidate_pool,
            number_of_outfits=3,
        )

        recommendation = validate_outfit_recommendation(
            recommendation=recommendation,
            source_item=source_item,
        )

    except Exception as error:
        recommendation = {
            "source_item_id": item_id,
            "source_item_type": source_type,
            "occasion": occasion,
            "summary": "Outfit recommendation failed.",
            "outfits": [],
            "error": str(error),
        }

    return recommendation


def print_outfit_recommendations(recommendation: dict) -> None:
    """
    Print outfit recommendations clearly.
    """
    print("\n" + "=" * 80)
    print("LLM OUTFIT RECOMMENDATIONS")
    print("=" * 80)

    print(f"\nSource item: {recommendation.get('source_item_id')}")
    print(f"Source item type: {recommendation.get('source_item_type')}")
    print(f"Occasion: {recommendation.get('occasion')}")
    print(f"Summary: {recommendation.get('summary', '')}")

    if recommendation.get("error"):
        print(f"Error: {recommendation.get('error')}")
        return

    outfits = recommendation.get("outfits", [])

    if not outfits:
        print("\nNo complete outfits generated.")
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