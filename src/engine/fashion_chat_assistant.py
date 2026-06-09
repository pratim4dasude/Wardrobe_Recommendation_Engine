import json
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.engine.chat_memory import ChatMemory
from src.engine.fashion_assistant import (
    find_similar_for_new_image,
    recommend_outfit_from_text_query,
)
from src.engine.outfit_recommender import (
    build_candidate_pool,
    detect_item_type,
    find_item_by_id,
    llm_create_outfit_recommendation,
    load_hybrid_items,
    validate_outfit_recommendation,
)
from src.engine.query_image_processor import process_query_image
from src.engine.query_understanding import understand_user_query
from src.engine.similar_search import (
    similar_search_by_item_id,
    similar_search_by_query_item,
)


load_dotenv()

client = OpenAI()

CHAT_MODEL = "gpt-4.1-mini"
HYBRID_EMBEDDINGS_PATH = "data/wardrobe_hybrid_embeddings.json"


def safe_json(data: dict | list) -> str:
    """
    Convert data to pretty JSON string.
    """
    return json.dumps(data, indent=2, ensure_ascii=False)


def extract_item_id(text: str) -> str | None:
    """
    Extract wardrobe item id like item_003.
    """
    match = re.search(r"\bitem_\d{3}\b", text.lower())

    if match:
        return match.group(0)

    return None


def extract_image_path(text: str) -> str | None:
    """
    Extract image path from user message.
    """
    pattern = (
        r"([A-Za-z]:\\[^\n\r]+?\.(?:jpg|jpeg|png|webp)"
        r"|[^\s]+?\.(?:jpg|jpeg|png|webp))"
    )
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip().strip('"').strip("'")

    return None


def infer_vibe_from_text(text: str | None) -> str | None:
    """
    Infer vibe from corrected query / occasion.
    Helps prevent old memory vibe from leaking.
    """
    if not text:
        return None

    text = text.lower()

    if "formal" in text:
        return "formal"

    if "smart casual" in text:
        return "smart casual"

    if "office" in text or "smart clean" in text or "clean" in text:
        return "clean smart casual"

    if "casual" in text or "day out" in text:
        return "casual"

    if "bold" in text:
        return "bold"

    if "classy" in text:
        return "classy"

    if "minimal" in text:
        return "minimal"

    if "street" in text:
        return "streetwear"

    if "party" in text:
        return "party"

    return None


def repair_parsed_intent(parsed: dict, user_message: str) -> dict:
    """
    Fix wrong intent caused by old memory context.

    Main fixes:
    - Fresh outfit requests should not use old image/item context.
    - item_id + occasion should directly style the item.
    - image_path + occasion should directly style the outside image.
    """
    text = user_message.lower()

    has_explicit_item = extract_item_id(user_message) is not None
    has_explicit_image = extract_image_path(user_message) is not None

    asks_similar = any(
        phrase in text
        for phrase in [
            "similar",
            "same",
            "like this",
            "what we have now",
            "matching item",
            "items like",
        ]
    )

    asks_new_outfit = any(
        phrase in text
        for phrase in [
            "what should i wear",
            "something to wear",
            "find something to wear",
            "help me find something to wear",
            "help to find something to wear",
            "outfit",
            "dress me",
        ]
    )

    # Fresh outfit request should not accidentally become outside_image_similar
    # because old image_path exists in memory.
    if (
        asks_new_outfit
        and not asks_similar
        and not has_explicit_item
        and not has_explicit_image
    ):
        parsed["intent"] = "text_outfit"
        parsed["tool_action"] = "recommend_outfit_from_text"
        parsed["item_id"] = None
        parsed["image_path"] = None
        parsed["is_followup"] = False
        parsed["refinement_request"] = None

        if "party" in text:
            parsed["occasion"] = "party"
        elif "office" in text:
            parsed["occasion"] = "office"
        elif "date" in text:
            parsed["occasion"] = "date"
        elif "wedding" in text:
            parsed["occasion"] = "wedding"
        elif "travel" in text:
            parsed["occasion"] = "travel"

        if not parsed.get("vibe"):
            parsed["needs_followup"] = True
            parsed["followup_question"] = (
                "What kind of vibe or style are you looking for? "
                "For example formal, casual, classy, bold, or smart clean?"
            )
        else:
            parsed["needs_followup"] = False
            parsed["followup_question"] = None

    # Explicit item styling should not ask extra vibe if occasion exists.
    if has_explicit_item:
        parsed["intent"] = "wardrobe_item_pairing"
        parsed["tool_action"] = "style_wardrobe_item"
        parsed["item_id"] = extract_item_id(user_message)
        parsed["image_path"] = None

        if parsed.get("occasion"):
            parsed["needs_followup"] = False
            parsed["followup_question"] = None

        if not parsed.get("vibe"):
            parsed["vibe"] = infer_vibe_from_text(
                parsed.get("corrected_query") or user_message
            )

    # Explicit outside image styling should not ask extra vibe if occasion exists.
    if has_explicit_image:
        parsed["image_path"] = extract_image_path(user_message)
        parsed["item_id"] = None

        if asks_similar:
            parsed["intent"] = "outside_image_similar"
            parsed["tool_action"] = "find_similar_for_outside_image"
        else:
            parsed["intent"] = "outside_image_pairing"
            parsed["tool_action"] = "style_outside_image"

        if parsed.get("occasion"):
            parsed["needs_followup"] = False
            parsed["followup_question"] = None

        if not parsed.get("vibe"):
            parsed["vibe"] = infer_vibe_from_text(
                parsed.get("corrected_query") or user_message
            )

    return parsed


def simplify_item(item: dict) -> dict:
    """
    Keep only useful fields for debug/user display.
    """
    return {
        "item_id": item.get("item_id"),
        "filename": item.get("filename"),
        "image_path": item.get("image_path"),
        "caption": item.get("caption"),
        "category": item.get("category", []),
        "color": item.get("color", []),
        "style": item.get("style", []),
        "item_type": detect_item_type(item),
        "bm25_score": item.get("bm25_score"),
        "hybrid_keyword_score": item.get("hybrid_keyword_score"),
    }


def build_outfit_for_source_item(
    source_item: dict,
    occasion: str,
    all_items: list[dict],
    number_of_outfits: int = 3,
) -> dict:
    """
    Create outfit recommendation for a given source garment.
    This is used by wardrobe item flow and outside image flow.
    """
    candidate_pool = build_candidate_pool(
        source_item=source_item,
        occasion=occasion,
        items=all_items,
        max_per_type=6,
    )

    recommendation = llm_create_outfit_recommendation(
        source_item=source_item,
        occasion=occasion,
        candidate_pool=candidate_pool,
        number_of_outfits=number_of_outfits,
    )

    recommendation = validate_outfit_recommendation(
        recommendation=recommendation,
        source_item=source_item,
    )

    return recommendation


def extract_first_outfit(recommendation: dict) -> dict | None:
    """
    Extract first outfit from recommendation.
    """
    outfits = recommendation.get("outfits", [])

    if not outfits:
        return None

    return outfits[0]


def build_alternative_outfits_from_similar_items(
    similar_items: list[dict],
    occasion: str,
    all_items: list[dict],
    max_items: int = 3,
) -> list[dict]:
    """
    For each similar wardrobe item, create one outfit.
    """
    alternative_outfits = []

    for index, similar_item in enumerate(similar_items[:max_items], start=1):
        similar_item_id = similar_item.get("item_id")
        print(f"Creating alternative outfit for similar item: {similar_item_id}")

        if not similar_item_id:
            continue

        try:
            similar_source_item = find_item_by_id(all_items, similar_item_id)

            recommendation = build_outfit_for_source_item(
                source_item=similar_source_item,
                occasion=occasion,
                all_items=all_items,
                number_of_outfits=1,
            )

            first_outfit = extract_first_outfit(recommendation)

            if first_outfit:
                first_outfit["outfit_rank"] = index
                first_outfit["similar_source_item_id"] = similar_item_id
                first_outfit["similar_source_caption"] = similar_source_item.get(
                    "caption", ""
                )
                alternative_outfits.append(first_outfit)

        except Exception as error:
            print(f"Could not build alternative outfit for {similar_item_id}: {error}")

    return alternative_outfits


def generate_user_intro(
    source_item: dict,
    occasion: str,
    present_in_wardrobe: bool,
    similar_items: list[dict],
) -> str:
    """
    Human stylist intro before structured outfit cards.
    """
    prompt = f"""
You are a friendly personal fashion stylist.

Source garment:
{safe_json({
    "item_id": source_item.get("item_id"),
    "filename": source_item.get("filename"),
    "caption": source_item.get("caption"),
    "category": source_item.get("category"),
    "color": source_item.get("color"),
    "style": source_item.get("style"),
    "present_in_wardrobe": present_in_wardrobe,
})}

Occasion/style request:
"{occasion}"

Similar wardrobe items found:
{safe_json([
    {
        "item_id": item.get("item_id"),
        "filename": item.get("filename"),
        "caption": item.get("caption"),
        "similarity_score": item.get("similarity_score"),
        "bm25_score": item.get("bm25_score"),
        "hybrid_keyword_score": item.get("hybrid_keyword_score"),
    }
    for item in similar_items[:3]
])}

Write a short natural intro before showing outfit cards.

Rules:
- Sound human and caring, not robotic.
- Mention whether the garment is already in the wardrobe or is an outside image.
- Say you will first style the exact input garment.
- Then say you also found similar wardrobe items for alternative looks.
- Keep it under 5 sentences.
- Do not use markdown headings.
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        input=prompt,
    )

    return response.output_text.strip()


def generate_stylist_note(
    source_item: dict,
    present_in_wardrobe: bool,
    selected_recommendation: dict,
    alternative_outfits: list[dict],
) -> str:
    """
    Final human stylist note for wardrobe item and outside image flow.
    """
    prompt = f"""
You are a friendly personal fashion stylist.

Source garment:
{safe_json({
    "item_id": source_item.get("item_id"),
    "caption": source_item.get("caption"),
    "present_in_wardrobe": present_in_wardrobe,
})}

Outfits using exact input garment:
{safe_json(selected_recommendation.get("outfits", []))}

Alternative outfits using similar wardrobe items:
{safe_json(alternative_outfits)}

Write a short final stylist note.

Rules:
- Mention which outfit you would personally pick first.
- If the source is outside wardrobe, mention whether buying/using it makes sense based on similar wardrobe items.
- Keep it under 4 sentences.
"""

    response = client.responses.create(
        model=CHAT_MODEL,
        input=prompt,
    )

    return response.output_text.strip()


def format_outfit_cards(recommendation: dict) -> str:
    """
    Format recommendation in structured outfit card style.
    """
    outfits = recommendation.get("outfits", [])

    if not outfits:
        return "No complete outfits generated."

    lines = []

    for outfit in outfits:
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"Outfit #{outfit.get('outfit_rank')}: {outfit.get('outfit_name')}")
        lines.append(f"Outfit type: {outfit.get('outfit_type')}")
        lines.append(f"Confidence: {outfit.get('confidence')}")
        lines.append(f"Overall reason: {outfit.get('overall_reason')}")
        lines.append(f"Styling notes: {outfit.get('styling_notes')}")
        lines.append("")
        lines.append("Items:")

        for item in outfit.get("items", []):
            lines.append(f"- Role: {item.get('role')}")
            lines.append(f"  Item ID: {item.get('item_id')}")
            lines.append(f"  Filename: {item.get('filename')}")
            lines.append(f"  Image path: {item.get('image_path')}")
            lines.append(f"  Reason: {item.get('reason')}")

    return "\n".join(lines)


def format_alternative_outfit_cards(alternative_outfits: list[dict]) -> str:
    """
    Format alternative outfits based on similar wardrobe garments.
    """
    if not alternative_outfits:
        return "No alternative outfits generated from similar wardrobe items."

    lines = []

    for outfit in alternative_outfits:
        lines.append("")
        lines.append("-" * 80)
        lines.append(
            f"Alternative Outfit #{outfit.get('outfit_rank')}: "
            f"{outfit.get('outfit_name')}"
        )
        lines.append(f"Based on similar item: {outfit.get('similar_source_item_id')}")
        lines.append(f"Similar item caption: {outfit.get('similar_source_caption')}")
        lines.append(f"Outfit type: {outfit.get('outfit_type')}")
        lines.append(f"Confidence: {outfit.get('confidence')}")
        lines.append(f"Overall reason: {outfit.get('overall_reason')}")
        lines.append(f"Styling notes: {outfit.get('styling_notes')}")
        lines.append("")
        lines.append("Items:")

        for item in outfit.get("items", []):
            lines.append(f"- Role: {item.get('role')}")
            lines.append(f"  Item ID: {item.get('item_id')}")
            lines.append(f"  Filename: {item.get('filename')}")
            lines.append(f"  Image path: {item.get('image_path')}")
            lines.append(f"  Reason: {item.get('reason')}")

    return "\n".join(lines)


def format_similar_breakdown(recommendation: dict) -> str:
    """
    Format similar item recommendation.
    """
    recommended_items = recommendation.get("recommended_items", [])

    if not recommended_items:
        return ""

    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("USER OUTPUT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(recommendation.get("summary", ""))

    for item in recommended_items:
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"Rank #{item.get('rank')}")
        lines.append(f"Item ID: {item.get('item_id')}")
        lines.append(f"Filename: {item.get('filename')}")
        lines.append(f"Image path: {item.get('image_path')}")
        lines.append(f"Match type: {item.get('match_type')}")
        lines.append(f"Reason: {item.get('reason')}")
        lines.append(f"Similarity explanation: {item.get('similarity_explanation')}")
        lines.append(f"Difference from source: {item.get('difference_from_source')}")

    return "\n".join(lines)


def format_text_outfit_response(recommendation: dict) -> str:
    """
    Format normal text-only outfit response.
    This keeps the previous structured output.
    """
    lines = []
    lines.append("")
    lines.append("=" * 80)
    lines.append("USER OUTPUT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(recommendation.get("summary", "Here are a few outfit ideas."))
    lines.append(format_outfit_cards(recommendation))

    return "\n".join(lines)


class FashionChatAssistant:
    """
    QnA style fashion assistant.

    This version keeps previous features and fixes:
    - fresh outfit requests after image search
    - unnecessary follow-up for item_id + occasion
    - refinement routing based on latest active source
    """

    def __init__(self, session_id: str | None = None):
        self.memory = ChatMemory(session_id=session_id)

    def classify_user_message(self, user_message: str) -> dict:
        """
        Query understanding layer.
        """
        context = self.memory.get_context()
        last_recommendation = self.memory.get_last_recommendation()

        parsed = understand_user_query(
            user_message=user_message,
            chat_context=context,
            has_previous_recommendation=bool(last_recommendation),
        )

        parsed = repair_parsed_intent(parsed, user_message)

        return parsed

    def merge_context(self, parsed: dict, user_message: str) -> dict:

        intent = parsed.get("intent")
        is_followup = parsed.get("is_followup", False)

        updates = {
            "raw_user_request": user_message,
            "last_user_request": parsed.get("corrected_query") or user_message,
        }

        is_new_request = (
                not is_followup
                and intent
                in [
                    "text_outfit",
                    "wardrobe_item_pairing",
                    "outside_image_pairing",
                    "outside_image_similar",
                ]
        )

        if is_new_request:
            updates.update(
                {
                    "occasion": None,
                    "vibe": None,
                    "color_preference": None,
                    "item_id": None,
                    "image_path": None,
                }
            )

        for key in [
            "occasion",
            "vibe",
            "color_preference",
            "item_id",
            "image_path",
        ]:
            if parsed.get(key):
                updates[key] = parsed.get(key)

        should_auto_infer_vibe = not (
                intent == "text_outfit"
                and parsed.get("needs_followup")
                and not parsed.get("vibe")
        )

        if not updates.get("vibe") and should_auto_infer_vibe:
            inferred_vibe = infer_vibe_from_text(
                parsed.get("corrected_query") or user_message
            )

            if inferred_vibe:
                updates["vibe"] = inferred_vibe

        if intent in [
            "text_outfit",
            "wardrobe_item_pairing",
            "outside_image_pairing",
            "outside_image_similar",
            "refine_previous",
        ]:
            updates["pending_action"] = intent

        if intent == "text_outfit":
            updates["last_active_source"] = "text"

        elif intent == "wardrobe_item_pairing":
            updates["last_active_source"] = "wardrobe_item"

        elif intent in ["outside_image_pairing", "outside_image_similar"]:
            updates["last_active_source"] = "outside_image"

        self.memory.update_context(updates)

        return self.memory.get_context()

    def build_followup_question(self, parsed: dict, context: dict) -> str | None:
        """
        Ask follow-up question only when needed.
        """
        intent = parsed.get("intent") or context.get("pending_action")

        occasion = context.get("occasion")
        vibe = context.get("vibe")
        item_id = context.get("item_id")
        image_path = context.get("image_path")

        ready_without_followup = (
            (intent == "wardrobe_item_pairing" and item_id and occasion)
            or (intent == "outside_image_pairing" and image_path and occasion)
            or (intent == "outside_image_similar" and image_path)
            or (intent == "text_outfit" and occasion and vibe)
        )

        if (
            parsed.get("needs_followup")
            and parsed.get("followup_question")
            and not ready_without_followup
        ):
            return parsed["followup_question"]

        if intent == "unclear":
            return (
                "Sure, I can help. Tell me the occasion, like party, office, date, "
                "casual outing, or share an item_id/image you want me to style."
            )

        if intent == "text_outfit":
            if not occasion:
                return (
                    "Sure, what is the occasion? For example: party, office, date, "
                    "casual outing, travel, or smart casual."
                )

            if not vibe:
                return (
                    f"Nice, I can suggest something for {occasion}. What vibe do you want: "
                    "minimal classy, bold statement, casual, streetwear, or semi-formal?"
                )

        if intent == "wardrobe_item_pairing":
            if not item_id:
                return (
                    "Sure, which wardrobe item do you want to pair? Send the item_id, "
                    "for example item_003."
                )

            if not occasion:
                return (
                    f"Got it. What occasion should I style {item_id} for? "
                    "For example party, office, smart casual outing, or casual everyday."
                )

            return None

        if intent == "outside_image_pairing":
            if not image_path:
                return (
                    "Sure, send the image path of the outside garment you want to pair, "
                    "for example C:\\Users\\KIIT\\...\\11.jpg"
                )

            if not occasion:
                return (
                    "Got the image. What occasion should I style it for: party, office, "
                    "casual outing, date, or smart casual?"
                )

            return None

        if intent == "outside_image_similar":
            if not image_path:
                return (
                    "Sure, send the image path and I will find similar items from your wardrobe."
                )

            return None

        return None

    def build_combined_query_from_context(self, context: dict) -> str:
        """
        Build clean query for recommendation tools.
        """
        parts = []

        if context.get("occasion"):
            parts.append(f"occasion: {context['occasion']}")

        if context.get("vibe"):
            parts.append(f"vibe: {context['vibe']}")

        if context.get("color_preference"):
            parts.append(f"color preference: {context['color_preference']}")

        if context.get("last_user_request"):
            parts.append(f"user request: {context['last_user_request']}")

        if not parts:
            return "casual everyday outfit"

        return ". ".join(parts)

    def build_wardrobe_item_styling_response(
        self,
        item_id: str,
        occasion: str,
    ) -> str:
        """
        Chat response for styling exact wardrobe item + similar alternatives.
        """
        debug_steps = []

        debug_steps.append(
            {
                "step": 1,
                "name": "Understanding request",
                "output": {
                    "intent": "wardrobe_item_pairing",
                    "item_id": item_id,
                    "occasion": occasion,
                },
            }
        )

        print("\nLoading wardrobe embeddings...")
        all_items = load_hybrid_items(HYBRID_EMBEDDINGS_PATH)

        print(f"Finding selected item: {item_id}")
        source_item = find_item_by_id(all_items, item_id)

        debug_steps.append(
            {
                "step": 2,
                "name": "Loading selected garment from wardrobe",
                "output": {
                    "present_in_wardrobe": True,
                    "source_item": simplify_item(source_item),
                    "text_embedding_available": bool(source_item.get("text_embedding")),
                    "visual_embedding_available": bool(
                        source_item.get("visual_embedding")
                    ),
                },
            }
        )

        print("\nCreating outfits using the selected garment...")
        selected_recommendation = build_outfit_for_source_item(
            source_item=source_item,
            occasion=occasion,
            all_items=all_items,
            number_of_outfits=3,
        )

        debug_steps.append(
            {
                "step": 3,
                "name": "Creating outfits using exact selected garment",
                "output": {
                    "generated_outfit_count": len(
                        selected_recommendation.get("outfits", [])
                    ),
                    "outfit_names": [
                        outfit.get("outfit_name")
                        for outfit in selected_recommendation.get("outfits", [])
                    ],
                },
            }
        )

        print("\nFinding similar wardrobe garments...")
        similar_items = similar_search_by_item_id(
            item_id=item_id,
            top_k=5,
            hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
        )

        debug_steps.append(
            {
                "step": 4,
                "name": "Finding similar garments in wardrobe",
                "output": [
                    {
                        "rank": index,
                        "item_id": item.get("item_id"),
                        "filename": item.get("filename"),
                        "similarity_score": item.get("similarity_score"),
                        "bm25_score": item.get("bm25_score"),
                        "hybrid_keyword_score": item.get("hybrid_keyword_score"),
                        "caption": item.get("caption"),
                    }
                    for index, item in enumerate(similar_items, start=1)
                ],
            }
        )

        print("\nCreating alternative outfits using similar garments...")
        alternative_outfits = build_alternative_outfits_from_similar_items(
            similar_items=similar_items,
            occasion=occasion,
            all_items=all_items,
            max_items=3,
        )

        debug_steps.append(
            {
                "step": 5,
                "name": "Creating alternative outfits using similar garments",
                "output": {
                    "alternative_outfit_count": len(alternative_outfits),
                    "based_on_similar_items": [
                        outfit.get("similar_source_item_id")
                        for outfit in alternative_outfits
                    ],
                },
            }
        )

        print("\nGenerating human stylist intro...")
        intro = generate_user_intro(
            source_item=source_item,
            occasion=occasion,
            present_in_wardrobe=True,
            similar_items=similar_items,
        )

        print("Generating final stylist note...")
        stylist_note = generate_stylist_note(
            source_item=source_item,
            present_in_wardrobe=True,
            selected_recommendation=selected_recommendation,
            alternative_outfits=alternative_outfits,
        )

        debug_steps.append(
            {
                "step": 6,
                "name": "Generating final user-facing response",
                "output": {
                    "human_intro_generated": True,
                    "structured_outfit_cards_generated": True,
                    "stylist_note_generated": True,
                },
            }
        )

        self.memory.set_last_recommendation(selected_recommendation)

        return self.format_full_chat_response(
            debug_steps=debug_steps,
            intro=intro,
            selected_title="OUTFITS USING YOUR SELECTED GARMENT",
            selected_recommendation=selected_recommendation,
            alternative_outfits=alternative_outfits,
            stylist_note=stylist_note,
        )

    def build_outside_image_styling_response(
        self,
        image_path: str | Path,
        occasion: str,
    ) -> str:
        """
        Chat response for styling outside image + similar wardrobe alternatives.
        """
        debug_steps = []

        debug_steps.append(
            {
                "step": 1,
                "name": "Understanding request",
                "output": {
                    "intent": "outside_image_pairing",
                    "image_path": str(image_path),
                    "occasion": occasion,
                },
            }
        )

        query_item = process_query_image(image_path)

        debug_steps.append(
            {
                "step": 2,
                "name": "Processing outside image as temporary garment",
                "output": {
                    "present_in_wardrobe": False,
                    "source_item": simplify_item(query_item),
                    "search_text": query_item.get("search_text"),
                    "text_embedding_created": bool(query_item.get("text_embedding")),
                    "text_vector_size": len(query_item.get("text_embedding", [])),
                    "visual_embedding_created": bool(
                        query_item.get("visual_embedding")
                    ),
                    "visual_vector_size": len(query_item.get("visual_embedding", [])),
                },
            }
        )

        all_items = load_hybrid_items(HYBRID_EMBEDDINGS_PATH)

        selected_recommendation = build_outfit_for_source_item(
            source_item=query_item,
            occasion=occasion,
            all_items=all_items,
            number_of_outfits=3,
        )

        debug_steps.append(
            {
                "step": 3,
                "name": "Creating outfits using exact outside input garment",
                "output": {
                    "generated_outfit_count": len(
                        selected_recommendation.get("outfits", [])
                    ),
                    "outfit_names": [
                        outfit.get("outfit_name")
                        for outfit in selected_recommendation.get("outfits", [])
                    ],
                },
            }
        )

        similar_items = similar_search_by_query_item(
            query_item=query_item,
            top_k=5,
            hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
        )

        debug_steps.append(
            {
                "step": 4,
                "name": "Finding similar garments already in wardrobe",
                "output": [
                    {
                        "rank": index,
                        "item_id": item.get("item_id"),
                        "filename": item.get("filename"),
                        "similarity_score": item.get("similarity_score"),
                        "text_score": item.get("text_score"),
                        "visual_score": item.get("visual_score"),
                        "bm25_score": item.get("bm25_score"),
                        "hybrid_keyword_score": item.get("hybrid_keyword_score"),
                        "caption": item.get("caption"),
                    }
                    for index, item in enumerate(similar_items, start=1)
                ],
            }
        )

        alternative_outfits = build_alternative_outfits_from_similar_items(
            similar_items=similar_items,
            occasion=occasion,
            all_items=all_items,
            max_items=3,
        )

        debug_steps.append(
            {
                "step": 5,
                "name": "Creating alternative outfits using similar wardrobe garments",
                "output": {
                    "alternative_outfit_count": len(alternative_outfits),
                    "based_on_similar_items": [
                        outfit.get("similar_source_item_id")
                        for outfit in alternative_outfits
                    ],
                },
            }
        )

        intro = generate_user_intro(
            source_item=query_item,
            occasion=occasion,
            present_in_wardrobe=False,
            similar_items=similar_items,
        )

        stylist_note = generate_stylist_note(
            source_item=query_item,
            present_in_wardrobe=False,
            selected_recommendation=selected_recommendation,
            alternative_outfits=alternative_outfits,
        )

        debug_steps.append(
            {
                "step": 6,
                "name": "Generating final user-facing response",
                "output": {
                    "human_intro_generated": True,
                    "structured_outfit_cards_generated": True,
                    "stylist_note_generated": True,
                },
            }
        )

        self.memory.set_last_recommendation(selected_recommendation)

        return self.format_full_chat_response(
            debug_steps=debug_steps,
            intro=intro,
            selected_title="OUTFITS USING YOUR INPUT GARMENT",
            selected_recommendation=selected_recommendation,
            alternative_outfits=alternative_outfits,
            stylist_note=stylist_note,
        )

    def format_full_chat_response(
        self,
        debug_steps: list[dict],
        intro: str,
        selected_title: str,
        selected_recommendation: dict,
        alternative_outfits: list[dict],
        stylist_note: str,
    ) -> str:
        """
        Build final chat message with:
        - system debug
        - user output
        """
        lines = []

        lines.append("=" * 80)
        lines.append("SYSTEM DEBUG / PIPELINE STEPS")
        lines.append("=" * 80)

        for step in debug_steps:
            lines.append("")
            lines.append(f"Step {step['step']}: {step['name']}")
            lines.append("-" * 80)
            lines.append(safe_json(step["output"]))

        lines.append("")
        lines.append("=" * 80)
        lines.append("USER OUTPUT")
        lines.append("=" * 80)
        lines.append("")
        lines.append(intro)

        lines.append("")
        lines.append("=" * 80)
        lines.append(selected_title)
        lines.append("=" * 80)
        lines.append(format_outfit_cards(selected_recommendation))

        lines.append("")
        lines.append("=" * 80)
        lines.append("ALTERNATIVE OUTFITS USING SIMILAR WARDROBE ITEMS")
        lines.append("=" * 80)
        lines.append(format_alternative_outfit_cards(alternative_outfits))

        lines.append("")
        lines.append("=" * 80)
        lines.append("STYLIST NOTE")
        lines.append("=" * 80)
        lines.append(stylist_note)

        return "\n".join(lines)

    def generate_text_human_response(
        self,
        recommendation: dict,
        user_message: str,
        context: dict,
    ) -> str:
        """
        Human response for normal text-only outfit flow.
        """
        prompt = f"""
You are a friendly personal fashion stylist.

User message:
"{user_message}"

Current context:
{safe_json(context)}

Recommendation result:
{safe_json(recommendation)}

Write a short natural human-style answer.

Rules:
- Mention the best option first.
- Explain why it works.
- Mention one styling tip.
- Do not include raw JSON.
- Keep it concise.
"""

        response = client.responses.create(
            model=CHAT_MODEL,
            input=prompt,
        )

        return response.output_text.strip()

    def generate_text_stylist_note(
        self,
        recommendation: dict,
        user_message: str,
        context: dict,
    ) -> str:
        """
        Final stylist note for normal text-only outfit flow.
        """
        prompt = f"""
You are a friendly personal fashion stylist.

User message:
"{user_message}"

Current context:
{safe_json(context)}

Recommendation result:
{safe_json(recommendation)}

Write a short final stylist note.

Rules:
- Mention which outfit you would personally pick first.
- Explain why it is the safest or best option.
- Give one small styling tip.
- Keep it under 4 sentences.
- Do not include raw JSON.
"""

        response = client.responses.create(
            model=CHAT_MODEL,
            input=prompt,
        )

        return response.output_text.strip()

    def run_text_outfit_response(self, user_message: str, context: dict) -> str:
        """
        Text-only outfit flow.
        """
        query = self.build_combined_query_from_context(context)

        recommendation = recommend_outfit_from_text_query(
            user_query=query,
            hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
        )

        self.memory.set_last_recommendation(recommendation)

        human_response = self.generate_text_human_response(
            recommendation=recommendation,
            user_message=user_message,
            context=context,
        )

        stylist_note = self.generate_text_stylist_note(
            recommendation=recommendation,
            user_message=user_message,
            context=context,
        )

        lines = []
        lines.append(human_response)
        lines.append(format_text_outfit_response(recommendation))
        lines.append("")
        lines.append("=" * 80)
        lines.append("STYLIST NOTE")
        lines.append("=" * 80)
        lines.append(stylist_note)

        return "\n\n".join(lines)

    def run_similar_image_response(self, image_path: str) -> str:
        """
        Similar search flow for outside image.
        """
        recommendation = find_similar_for_new_image(
            image_path=image_path,
            hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
        )

        self.memory.set_last_recommendation(recommendation)

        return format_similar_breakdown(recommendation)

    def chat(self, user_message: str) -> str:
        """
        Main chat function.
        """
        self.memory.add_message("user", user_message)

        parsed = self.classify_user_message(user_message)

        print("\n" + "=" * 80)
        print("QUERY UNDERSTANDING LAYER")
        print("=" * 80)
        print(safe_json(parsed))

        context = self.merge_context(parsed, user_message)

        followup_question = self.build_followup_question(parsed, context)

        if followup_question:
            self.memory.add_message("assistant", followup_question)
            return followup_question

        intent = context.get("pending_action") or parsed.get("intent") or "text_outfit"
        query = self.build_combined_query_from_context(context)

        print("\n" + "=" * 80)
        print("FINAL CLEAN QUERY SENT TO RECOMMENDATION SYSTEM")
        print("=" * 80)
        print(query)

        if intent == "wardrobe_item_pairing":
            assistant_response = self.build_wardrobe_item_styling_response(
                item_id=context["item_id"],
                occasion=query,
            )

        elif intent == "outside_image_pairing":
            assistant_response = self.build_outside_image_styling_response(
                image_path=context["image_path"],
                occasion=query,
            )

        elif intent == "outside_image_similar":
            assistant_response = self.run_similar_image_response(
                image_path=context["image_path"],
            )

        elif intent == "refine_previous":
            last_active_source = context.get("last_active_source")

            if last_active_source == "outside_image" and context.get("image_path"):
                assistant_response = self.build_outside_image_styling_response(
                    image_path=context["image_path"],
                    occasion=query,
                )

            elif last_active_source == "wardrobe_item" and context.get("item_id"):
                assistant_response = self.build_wardrobe_item_styling_response(
                    item_id=context["item_id"],
                    occasion=query,
                )

            else:
                assistant_response = self.run_text_outfit_response(
                    user_message=user_message,
                    context=context,
                )

        else:
            assistant_response = self.run_text_outfit_response(
                user_message=user_message,
                context=context,
            )

        self.memory.add_message("assistant", assistant_response)

        return assistant_response