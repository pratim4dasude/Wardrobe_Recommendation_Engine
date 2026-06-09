import json
import math
from collections import defaultdict

from dotenv import load_dotenv
from openai import OpenAI

from src.engine.bm25_retrieval import rerank_items_with_bm25
from src.engine.embeddings import create_text_embedding


load_dotenv()

client = OpenAI()

OUTFIT_MODEL = "gpt-4.1-mini"
HYBRID_EMBEDDINGS_PATH = "data/wardrobe_hybrid_embeddings.json"


def make_json_safe(value):
    """
    Convert numpy/float32/float64 values into normal Python JSON-safe values.

    This prevents:
    Object of type float32 is not JSON serializable
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, set):
        return [make_json_safe(item) for item in value]

    if hasattr(value, "item"):
        return make_json_safe(value.item())

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def safe_float(value, default: float = 0.0) -> float:
    """
    Convert Python/numpy numeric values into normal Python float.
    """
    if value is None:
        return default

    if hasattr(value, "item"):
        value = value.item()

    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_hybrid_items(hybrid_embeddings_path: str | None = None) -> list[dict]:
    """
    Load all wardrobe items from pgvector DB.

    hybrid_embeddings_path is kept only so old function calls do not break.
    """
    from src.db.pgvector_store import load_all_items

    items = load_all_items()

    if not items:
        raise ValueError(
            "No wardrobe items found in pgvector DB. "
            "Run: python scripts\\migrate_json_to_pgvector.py"
        )

    print(f"Loaded {len(items)} wardrobe items from pgvector DB.")

    return make_json_safe(items)


def find_item_by_id(items: list[dict], item_id: str) -> dict:
    """
    Find source wardrobe item by item_id.
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

    vector_a = [safe_float(value) for value in vector_a]
    vector_b = [safe_float(value) for value in vector_b]

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=False))
    norm_a = math.sqrt(sum(a * a for a in vector_a))
    norm_b = math.sqrt(sum(b * b for b in vector_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(dot_product / (norm_a * norm_b))


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
        for word in [
            "jeans",
            "trousers",
            "pants",
            "shorts",
            "bottomwear",
            "bottom wear",
        ]
    ):
        return "bottomwear"

    if any(
        word in full_text
        for word in [
            "sneakers",
            "shoes",
            "footwear",
            "sandals",
            "boots",
            "loafers",
        ]
    ):
        return "footwear"

    if any(
        word in full_text
        for word in [
            "jacket",
            "blazer",
            "coat",
            "outerwear",
            "hoodie",
            "sweater",
            "cardigan",
        ]
    ):
        return "outerwear"

    if any(word in full_text for word in ["dress", "gown"]):
        return "dress"

    if any(
        word in full_text
        for word in [
            "accessory",
            "bag",
            "belt",
            "watch",
            "cap",
            "hat",
            "scarf",
        ]
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
            "button-up",
            "button-down",
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


def normalize_text_list(values: list | str | None) -> set[str]:
    """
    Convert style/category/color metadata into lowercase set.
    """
    if not values:
        return set()

    if isinstance(values, str):
        return {values.lower()}

    return {str(value).lower() for value in values}


def metadata_bonus_score(source_item: dict, candidate_item: dict, occasion: str) -> float:
    """
    Rule-based score to help candidate selection before LLM.
    """
    score = 0.0

    source_styles = normalize_text_list(source_item.get("style", []))
    candidate_styles = normalize_text_list(candidate_item.get("style", []))

    source_colors = normalize_text_list(source_item.get("color", []))
    candidate_colors = normalize_text_list(candidate_item.get("color", []))

    source_categories = normalize_text_list(source_item.get("category", []))
    candidate_categories = normalize_text_list(candidate_item.get("category", []))

    style_overlap = source_styles.intersection(candidate_styles)
    color_overlap = source_colors.intersection(candidate_colors)
    category_overlap = source_categories.intersection(candidate_categories)

    score += len(style_overlap) * 0.05
    score += len(color_overlap) * 0.03
    score += len(category_overlap) * 0.02

    occasion_text = occasion.lower()
    candidate_text = (
        " ".join(candidate_item.get("style", []))
        + " "
        + " ".join(candidate_item.get("category", []))
        + " "
        + " ".join(candidate_item.get("color", []))
        + " "
        + candidate_item.get("caption", "")
        + " "
        + candidate_item.get("search_text", "")
    ).lower()

    occasion_keywords = [
        "formal",
        "office",
        "party",
        "casual",
        "smart",
        "clean",
        "classy",
        "streetwear",
        "wedding",
        "travel",
        "date",
        "outing",
        "blazer",
        "trousers",
        "denim",
        "plaid",
        "flannel",
    ]

    for keyword in occasion_keywords:
        if keyword in occasion_text and keyword in candidate_text:
            score += 0.05

    for word in occasion_text.split():
        if word in candidate_text:
            score += 0.02

    return round(float(score), 4)


def build_outfit_query(source_item: dict, occasion: str) -> str:
    """
    Build query used for text embedding retrieval and BM25 scoring.
    """
    return (
        f"Outfit item compatible with: {source_item.get('caption', '')}. "
        f"Occasion/style intent: {occasion}. "
        f"Source category: {source_item.get('category', [])}. "
        f"Source colors: {source_item.get('color', [])}. "
        f"Source style: {source_item.get('style', [])}. "
        f"Source search text: {source_item.get('search_text', '')}."
    )


def compact_candidate_for_llm(item: dict) -> dict:
    """
    Keep only useful fields for the LLM prompt.
    This avoids sending unnecessary large data.
    """
    compact_item = {
        "item_id": item.get("item_id"),
        "filename": item.get("filename"),
        "image_path": item.get("image_path"),
        "item_type": item.get("item_type"),
        "caption": item.get("caption", ""),
        "category": item.get("category", []),
        "color": item.get("color", []),
        "style": item.get("style", []),
        "search_text": item.get("search_text", ""),
        "text_score": safe_float(item.get("text_score")),
        "metadata_score": safe_float(item.get("metadata_score")),
        "candidate_score": safe_float(item.get("candidate_score")),
        "bm25_score": safe_float(item.get("bm25_score")),
        "bm25_raw_score": safe_float(item.get("bm25_raw_score")),
        "hybrid_keyword_score": safe_float(item.get("hybrid_keyword_score")),
    }

    return make_json_safe(compact_item)


# def build_candidate_pool(
#     source_item: dict,
#     occasion: str,
#     items: list[dict],
#     max_per_type: int = 6,
# ) -> dict[str, list[dict]]:
#     """
#     Retrieve compatible candidates grouped by outfit role/type.
#
#     Ranking layers:
#     1. Text embedding similarity
#     2. Metadata/style/category bonus
#     3. BM25 keyword score
#     4. Final hybrid keyword score
#     """
#     source_type = detect_item_type(source_item)
#     target_types = get_target_item_types(source_type)
#
#     outfit_query = build_outfit_query(source_item=source_item, occasion=occasion)
#     query_embedding = create_text_embedding(outfit_query)
#     query_embedding = [safe_float(value) for value in query_embedding]
#
#     grouped_candidates = defaultdict(list)
#
#     for item in items:
#         if item.get("item_id") == source_item.get("item_id"):
#             continue
#
#         candidate_type = detect_item_type(item)
#
#         if candidate_type not in target_types:
#             continue
#
#         if not item.get("text_embedding"):
#             continue
#
#         text_embedding = [safe_float(value) for value in item["text_embedding"]]
#
#         text_score = safe_float(cosine_similarity(query_embedding, text_embedding))
#         metadata_score = safe_float(metadata_bonus_score(source_item, item, occasion))
#         candidate_score = safe_float(text_score + metadata_score)
#
#         candidate = {
#             "item_id": item.get("item_id"),
#             "filename": item.get("filename"),
#             "image_path": item.get("image_path"),
#             "item_type": candidate_type,
#             "candidate_score": round(candidate_score, 4),
#             "similarity_score": round(candidate_score, 4),
#             "text_score": round(text_score, 4),
#             "metadata_score": round(metadata_score, 4),
#             "caption": item.get("caption", ""),
#             "category": item.get("category", []),
#             "color": item.get("color", []),
#             "style": item.get("style", []),
#             "search_text": item.get("search_text", ""),
#         }
#
#         grouped_candidates[candidate_type].append(make_json_safe(candidate))
#
#     final_grouped_candidates = {}
#
#     for item_type, candidates in grouped_candidates.items():
#         candidates = sorted(
#             candidates,
#             key=lambda item: safe_float(item.get("candidate_score")),
#             reverse=True,
#         )
#
#         shortlist_size = max(max_per_type * 3, max_per_type)
#         candidates = candidates[:shortlist_size]
#
#         bm25_reranked_candidates = rerank_items_with_bm25(
#             query=outfit_query,
#             items=candidates,
#             bm25_weight=0.30,
#             existing_score_weight=0.70,
#         )
#
#         bm25_reranked_candidates = make_json_safe(bm25_reranked_candidates)
#
#         print("\n" + "=" * 80)
#         print(f"BM25 + HYBRID RERANKING DEBUG | item_type: {item_type}")
#         print("=" * 80)
#         print(f"BM25 query: {outfit_query[:300]}...")
#
#         for rank, candidate in enumerate(bm25_reranked_candidates[:5], start=1):
#             print(
#                 f"{rank}. {candidate.get('item_id')} | "
#                 f"text_score={candidate.get('text_score')} | "
#                 f"metadata_score={candidate.get('metadata_score')} | "
#                 f"candidate_score={candidate.get('candidate_score')} | "
#                 f"bm25_score={candidate.get('bm25_score')} | "
#                 f"hybrid_keyword_score={candidate.get('hybrid_keyword_score')}"
#             )
#             print(f"   caption: {candidate.get('caption', '')[:160]}")
#
#         final_grouped_candidates[item_type] = [
#             compact_candidate_for_llm(item)
#             for item in bm25_reranked_candidates[:max_per_type]
#         ]
#
#     return make_json_safe(final_grouped_candidates)

def build_candidate_pool(
    source_item: dict,
    occasion: str,
    items: list[dict],
    max_per_type: int = 6,
) -> dict[str, list[dict]]:

    from src.db.pgvector_store import search_by_text_embedding

    source_type = detect_item_type(source_item)
    target_types = get_target_item_types(source_type)

    outfit_query = build_outfit_query(source_item=source_item, occasion=occasion)
    query_embedding = create_text_embedding(outfit_query)
    query_embedding = [safe_float(value) for value in query_embedding]

    final_grouped_candidates = {}

    for item_type in target_types:
        pgvector_candidates = search_by_text_embedding(
            query_embedding=query_embedding,
            top_k=max_per_type * 3,
            item_type=item_type,
            exclude_item_id=source_item.get("item_id"),
        )

        if not pgvector_candidates:
            continue

        candidates = []

        for item in pgvector_candidates:
            text_score = safe_float(item.get("text_score"))
            metadata_score = safe_float(
                metadata_bonus_score(source_item, item, occasion)
            )
            candidate_score = safe_float(text_score + metadata_score)

            candidate = {
                "item_id": item.get("item_id"),
                "filename": item.get("filename"),
                "image_path": item.get("image_path"),
                "item_type": item_type,
                "candidate_score": round(candidate_score, 4),
                "similarity_score": round(candidate_score, 4),
                "text_score": round(text_score, 4),
                "metadata_score": round(metadata_score, 4),
                "caption": item.get("caption", ""),
                "category": item.get("category", []),
                "color": item.get("color", []),
                "style": item.get("style", []),
                "search_text": item.get("search_text", ""),
            }

            candidates.append(make_json_safe(candidate))

        candidates = sorted(
            candidates,
            key=lambda item: safe_float(item.get("candidate_score")),
            reverse=True,
        )

        bm25_reranked_candidates = rerank_items_with_bm25(
            query=outfit_query,
            items=candidates,
            bm25_weight=0.30,
            existing_score_weight=0.70,
        )

        bm25_reranked_candidates = make_json_safe(bm25_reranked_candidates)

        print("\n" + "=" * 80)
        print(f"PGVECTOR + BM25 RERANKING DEBUG | item_type: {item_type}")
        print("=" * 80)
        print(f"PGVECTOR query: {outfit_query[:300]}...")

        for rank, candidate in enumerate(bm25_reranked_candidates[:5], start=1):
            print(
                f"{rank}. {candidate.get('item_id')} | "
                f"text_score={candidate.get('text_score')} | "
                f"metadata_score={candidate.get('metadata_score')} | "
                f"candidate_score={candidate.get('candidate_score')} | "
                f"bm25_score={candidate.get('bm25_score')} | "
                f"hybrid_keyword_score={candidate.get('hybrid_keyword_score')}"
            )
            print(f"   caption: {candidate.get('caption', '')[:160]}")

        final_grouped_candidates[item_type] = [
            compact_candidate_for_llm(item)
            for item in bm25_reranked_candidates[:max_per_type]
        ]

    return make_json_safe(final_grouped_candidates)


def build_source_context(source_item: dict) -> dict:
    """
    Prepare source item for LLM.
    """
    source_context = {
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

    return make_json_safe(source_context)


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
    source_item = make_json_safe(source_item)
    candidate_pool = make_json_safe(candidate_pool)

    source_context = build_source_context(source_item)

    source_type = detect_item_type(source_item)
    required_roles = get_required_roles_for_source_type(source_type)

    source_context_json = json.dumps(
        make_json_safe(source_context),
        indent=2,
        ensure_ascii=False,
    )

    required_roles_json = json.dumps(
        make_json_safe(required_roles),
        indent=2,
        ensure_ascii=False,
    )

    candidate_pool_json = json.dumps(
        make_json_safe(candidate_pool),
        indent=2,
        ensure_ascii=False,
    )

    prompt = f"""
You are a fashion outfit recommendation assistant.

The user selected this source wardrobe item:
{source_context_json}

Detected source item type:
"{source_type}"

Required roles for a complete outfit:
{required_roles_json}

The user wants an outfit for this occasion/style intent:
"{occasion}"

Below are compatible wardrobe candidates retrieved from the user's closet.
They are grouped by item type:
{candidate_pool_json}

Candidate scoring meaning:
- text_score: semantic compatibility using text embeddings.
- metadata_score: simple style/category/color/occasion bonus.
- bm25_score: exact keyword match score for words like plaid, flannel, formal, office, denim, blazer, trousers.
- hybrid_keyword_score: combined retrieval score using semantic + keyword signals.
Use these scores as guidance, but make the final decision based on wearability, occasion fit, color harmony, and outfit completeness.

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
    """
    recommendation = make_json_safe(recommendation)
    source_item = make_json_safe(source_item)

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

        if source_type == "upperwear" and "bottomwear" not in roles:
            continue

        if source_type == "bottomwear" and "upperwear" not in roles:
            continue

        if source_type in ["outerwear", "footwear", "accessory"]:
            if "upperwear" not in roles or "bottomwear" not in roles:
                continue

        valid_outfits.append(outfit)

    recommendation["outfits"] = valid_outfits

    if not valid_outfits:
        recommendation["summary"] = (
            recommendation.get("summary", "")
            + " No complete outfits passed validation."
        )

    return make_json_safe(recommendation)


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

    source_item = make_json_safe(source_item)

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

    candidate_pool = make_json_safe(candidate_pool)

    for item_type, candidates in candidate_pool.items():
        print(f"{item_type}: {len(candidates)} candidate(s)")

        for index, candidate in enumerate(candidates[:3], start=1):
            print(
                f"  {index}. {candidate.get('item_id')} | "
                f"text={candidate.get('text_score')} | "
                f"bm25={candidate.get('bm25_score')} | "
                f"hybrid={candidate.get('hybrid_keyword_score')}"
            )

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

    return make_json_safe(recommendation)


def print_outfit_recommendations(recommendation: dict) -> None:
    """
    Print outfit recommendations clearly.
    """
    recommendation = make_json_safe(recommendation)

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