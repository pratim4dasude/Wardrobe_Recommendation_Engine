import json
import re

from dotenv import load_dotenv
from openai import OpenAI

from src.engine.chat_memory import ChatMemory
from src.engine.fashion_assistant import (
    find_similar_for_new_image,
    recommend_outfit_from_new_image,
    recommend_outfit_from_text_query,
    recommend_outfit_from_wardrobe_item,
)


load_dotenv()

client = OpenAI()

CHAT_MODEL = "gpt-4.1-mini"
HYBRID_EMBEDDINGS_PATH = "data/wardrobe_hybrid_embeddings.json"


def extract_json_from_text(text: str) -> dict:
    """
    Safely extract JSON from LLM output.
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

    Supports:
    - C:\\Users\\...\\11.jpg
    - data/new_query_images/11.jpg
    - /home/.../11.jpg
    """
    pattern = r"([A-Za-z]:\\[^\n\r]+?\.(?:jpg|jpeg|png|webp)|[^\s]+?\.(?:jpg|jpeg|png|webp))"
    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip().strip('"').strip("'")

    return None


def format_outfit_breakdown(recommendation: dict) -> str:
    """
    Convert outfit recommendation JSON into readable terminal format.
    """
    outfits = recommendation.get("outfits", [])

    if not outfits:
        return ""

    lines = []
    lines.append("")
    lines.append("Detailed outfit breakdown:")
    lines.append("=" * 80)

    source_item_id = recommendation.get("source_item_id")
    source_item_type = recommendation.get("source_item_type")
    occasion = recommendation.get("occasion")
    user_query = recommendation.get("user_query")

    if source_item_id:
        lines.append(f"Source item: {source_item_id}")

    if source_item_type:
        lines.append(f"Source item type: {source_item_type}")

    if occasion:
        lines.append(f"Occasion: {occasion}")

    if user_query:
        lines.append(f"User query: {user_query}")

    summary = recommendation.get("summary", "")
    if summary:
        lines.append(f"Summary: {summary}")

    for outfit in outfits:
        lines.append("")
        lines.append("-" * 80)
        lines.append(
            f"Outfit #{outfit.get('outfit_rank')}: {outfit.get('outfit_name')}"
        )
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
    Convert similar item recommendation JSON into readable terminal format.
    """
    recommended_items = recommendation.get("recommended_items", [])

    if not recommended_items:
        return ""

    lines = []
    lines.append("")
    lines.append("Detailed similar item breakdown:")
    lines.append("=" * 80)

    summary = recommendation.get("summary", "")
    if summary:
        lines.append(f"Summary: {summary}")

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


class FashionChatAssistant:
    """
    QnA style fashion assistant.

    It can:
    - ask follow-up questions
    - remember previous context
    - recommend outfits from text
    - pair wardrobe item
    - pair outside image
    - find similar wardrobe items from outside image
    """

    def __init__(self, session_id: str | None = None):
        self.memory = ChatMemory(session_id=session_id)

    def classify_user_message(self, user_message: str) -> dict:
        """
        Use LLM to extract intent and fashion slots.
        """
        context = self.memory.get_context()
        last_recommendation = self.memory.get_last_recommendation()

        prompt = f"""
You are an intent parser for a fashion assistant.

Current chat context:
{json.dumps(context, indent=2, ensure_ascii=False)}

Does the user have a previous recommendation?
{bool(last_recommendation)}

User message:
"{user_message}"

Extract the user's fashion intent.

Possible intent values:
- text_outfit
- wardrobe_item_pairing
- outside_image_pairing
- outside_image_similar
- refine_previous
- unclear

Rules:
- If user asks what to wear without item/image, use text_outfit.
- If user mentions item_id like item_003 and asks pair/wear/style, use wardrobe_item_pairing.
- If user mentions an image path and asks pair/outfit/wear, use outside_image_pairing.
- If user mentions an image path and asks similar/same/like this, use outside_image_similar.
- If user says make it casual/formal/bold/change color/another option and previous recommendation exists, use refine_previous.
- Extract occasion if mentioned, like party, office, date, casual outing, wedding, travel, smart casual.
- Extract vibe if mentioned, like bold, minimal, classy, streetwear, formal, casual, semi-formal.
- Extract color preference if mentioned, like dark, black, blue, colorful, neutral.
- Return only valid JSON.

Return this JSON structure:
{{
  "intent": "",
  "occasion": null,
  "vibe": null,
  "color_preference": null,
  "item_id": null,
  "image_path": null,
  "needs_followup": false,
  "followup_reason": "",
  "refinement_request": null
}}
"""

        response = client.responses.create(
            model=CHAT_MODEL,
            input=prompt,
        )

        raw_text = response.output_text.strip()

        try:
            parsed = extract_json_from_text(raw_text)
        except Exception:
            parsed = {
                "intent": "unclear",
                "occasion": None,
                "vibe": None,
                "color_preference": None,
                "item_id": None,
                "image_path": None,
                "needs_followup": True,
                "followup_reason": "Could not understand the request clearly.",
                "refinement_request": None,
            }

        detected_item_id = extract_item_id(user_message)
        detected_image_path = extract_image_path(user_message)

        if detected_item_id:
            parsed["item_id"] = detected_item_id

        if detected_image_path:
            parsed["image_path"] = detected_image_path

        return parsed

    def merge_context(self, parsed: dict, user_message: str) -> dict:
        """
        Merge parsed data into memory context.
        """
        updates = {
            "last_user_request": user_message,
        }

        for key in ["occasion", "vibe", "color_preference", "item_id", "image_path"]:
            if parsed.get(key):
                updates[key] = parsed.get(key)

        intent = parsed.get("intent")

        if intent in [
            "text_outfit",
            "wardrobe_item_pairing",
            "outside_image_pairing",
            "outside_image_similar",
            "refine_previous",
        ]:
            updates["pending_action"] = intent

        self.memory.update_context(updates)

        return self.memory.get_context()

    def build_followup_question(self, parsed: dict, context: dict) -> str | None:
        """
        Ask smart question if important information is missing.
        """
        intent = parsed.get("intent") or context.get("pending_action")

        occasion = context.get("occasion")
        vibe = context.get("vibe")
        item_id = context.get("item_id")
        image_path = context.get("image_path")

        if intent == "unclear":
            return (
                "Sure, I can help. Are you looking for a full outfit, something to pair "
                "with a wardrobe item, or similar items from an image?"
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

        if intent == "outside_image_similar":
            if not image_path:
                return (
                    "Sure, send the image path and I will find similar items from your wardrobe."
                )

        return None

    def build_combined_query_from_context(self, context: dict) -> str:
        """
        Build a clean query for recommendation tools.
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

    def generate_human_response(
        self,
        recommendation: dict,
        user_message: str,
        context: dict,
    ) -> str:
        """
        Convert structured recommendation into natural stylist-style response.
        """
        prompt = f"""
You are a friendly personal fashion stylist.

User message:
"{user_message}"

Current context:
{json.dumps(context, indent=2, ensure_ascii=False)}

Recommendation result:
{json.dumps(recommendation, indent=2, ensure_ascii=False)}

Write a natural human-style answer.

Style:
- Sound like a helpful stylist.
- Keep it short.
- Mention the best option first.
- Explain why it works.
- Mention 1 small styling tip.
- If there are multiple outfits, summarize them clearly.
- Do not output raw JSON.
- Do not include the detailed item breakdown because it will be added separately.
"""

        response = client.responses.create(
            model=CHAT_MODEL,
            input=prompt,
        )

        return response.output_text.strip()

    def run_recommendation(self, intent: str, user_message: str, context: dict) -> dict:
        """
        Call the correct backend tool.
        """
        query = self.build_combined_query_from_context(context)

        if intent == "refine_previous":
            refine_query = (
                f"Previous user request: {context.get('last_user_request')}. "
                f"New refinement: {user_message}. "
                f"Use this preference: {query}."
            )

            if context.get("item_id"):
                return recommend_outfit_from_wardrobe_item(
                    item_id=context["item_id"],
                    user_query=refine_query,
                    hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
                )

            if context.get("image_path"):
                return recommend_outfit_from_new_image(
                    image_path=context["image_path"],
                    user_query=refine_query,
                    hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
                )

            return recommend_outfit_from_text_query(
                user_query=refine_query,
                hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
            )

        if intent == "wardrobe_item_pairing":
            return recommend_outfit_from_wardrobe_item(
                item_id=context["item_id"],
                user_query=query,
                hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
            )

        if intent == "outside_image_pairing":
            return recommend_outfit_from_new_image(
                image_path=context["image_path"],
                user_query=query,
                hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
            )

        if intent == "outside_image_similar":
            return find_similar_for_new_image(
                image_path=context["image_path"],
                hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
            )

        return recommend_outfit_from_text_query(
            user_query=query,
            hybrid_embeddings_path=HYBRID_EMBEDDINGS_PATH,
        )

    def build_structured_breakdown(self, recommendation: dict) -> str:
        """
        Add structured outfit/similar result details after human response.
        """
        if recommendation.get("recommended_items"):
            return format_similar_breakdown(recommendation)

        if recommendation.get("outfits"):
            return format_outfit_breakdown(recommendation)

        return ""

    def chat(self, user_message: str) -> str:
        """
        Main chat function.
        """
        self.memory.add_message("user", user_message)

        parsed = self.classify_user_message(user_message)
        context = self.merge_context(parsed, user_message)

        followup_question = self.build_followup_question(parsed, context)

        if followup_question:
            self.memory.add_message("assistant", followup_question)
            return followup_question

        intent = context.get("pending_action") or parsed.get("intent") or "text_outfit"

        recommendation = self.run_recommendation(
            intent=intent,
            user_message=user_message,
            context=context,
        )

        self.memory.set_last_recommendation(recommendation)

        human_response = self.generate_human_response(
            recommendation=recommendation,
            user_message=user_message,
            context=context,
        )

        structured_breakdown = self.build_structured_breakdown(recommendation)

        if structured_breakdown:
            assistant_response = f"{human_response}\n\n{structured_breakdown}"
        else:
            assistant_response = human_response

        self.memory.add_message("assistant", assistant_response)

        return assistant_response