import json
import re

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()

QUERY_UNDERSTANDING_MODEL = "gpt-4.1-mini"


def safe_json(data: dict | list) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


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
    pattern = (
        r"([A-Za-z]:\\[^\n\r]+?\.(?:jpg|jpeg|png|webp)"
        r"|[^\s]+?\.(?:jpg|jpeg|png|webp))"
    )

    match = re.search(pattern, text, re.IGNORECASE)

    if match:
        return match.group(1).strip().strip('"').strip("'")

    return None


def normalize_common_typos(text: str) -> str:
    """
    Light typo correction before sending to LLM.
    """
    normalized = text.lower().strip()

    replacements = {
        "forma": "formal",
        "forml": "formal",
        "part ": "party ",
        " part": " party",
        "prt": "party",
        "ofice": "office",
        "offce": "office",
        "casule": "casual",
        "causal": "casual",
        "minmal": "minimal",
        "clas": "classy",
        "classi": "classy",
        "semiformal": "semi-formal",
        "ware": "wear",
        "wer": "wear",
        "waer": "wear",
        "outift": "outfit",
        "suggent": "suggest",
        "sugget": "suggest",
        "paird": "pair",
    }

    for wrong, correct in replacements.items():
        normalized = normalized.replace(wrong, correct)

    return normalized


def infer_followup_without_llm(
    user_message: str,
    chat_context: dict,
) -> dict | None:
    """
    If assistant has already asked a follow-up question,
    treat next user message as the missing answer.
    """
    normalized = normalize_common_typos(user_message)

    pending_action = chat_context.get("pending_action")
    occasion = chat_context.get("occasion")
    vibe = chat_context.get("vibe")
    item_id = chat_context.get("item_id")
    image_path = chat_context.get("image_path")

    detected_item_id = extract_item_id(user_message)
    detected_image_path = extract_image_path(user_message)

    if detected_item_id or detected_image_path:
        return None

    if pending_action == "text_outfit" and occasion and not vibe:
        inferred_vibe = normalized
        color_preference = None

        if "formal" in normalized:
            inferred_vibe = "formal"
        elif "classy" in normalized:
            inferred_vibe = "classy"
        elif "bold" in normalized:
            inferred_vibe = "bold"
        elif "minimal" in normalized:
            inferred_vibe = "minimal"
        elif "casual" in normalized:
            inferred_vibe = "casual"
        elif "street" in normalized:
            inferred_vibe = "streetwear"
        elif "clean" in normalized or "smart" in normalized:
            inferred_vibe = "clean smart casual"

        if "dark" in normalized or "black" in normalized:
            color_preference = "dark"
        elif "blue" in normalized:
            color_preference = "blue"
        elif "neutral" in normalized:
            color_preference = "neutral"

        return {
            "raw_query": user_message,
            "corrected_query": normalized,
            "intent": "text_outfit",
            "tool_action": "recommend_outfit_from_text",
            "occasion": occasion,
            "vibe": inferred_vibe,
            "color_preference": color_preference,
            "item_id": None,
            "image_path": None,
            "is_followup": True,
            "needs_followup": False,
            "followup_question": None,
            "refinement_request": None,
        }

    if pending_action == "wardrobe_item_pairing" and item_id and not occasion:
        return {
            "raw_query": user_message,
            "corrected_query": normalized,
            "intent": "wardrobe_item_pairing",
            "tool_action": "style_wardrobe_item",
            "occasion": normalized,
            "vibe": None,
            "color_preference": None,
            "item_id": item_id,
            "image_path": None,
            "is_followup": True,
            "needs_followup": False,
            "followup_question": None,
            "refinement_request": None,
        }

    if pending_action == "outside_image_pairing" and image_path and not occasion:
        return {
            "raw_query": user_message,
            "corrected_query": normalized,
            "intent": "outside_image_pairing",
            "tool_action": "style_outside_image",
            "occasion": normalized,
            "vibe": None,
            "color_preference": None,
            "item_id": None,
            "image_path": image_path,
            "is_followup": True,
            "needs_followup": False,
            "followup_question": None,
            "refinement_request": None,
        }

    return None


def fallback_understanding(user_message: str, chat_context: dict) -> dict:
    """
    Rule-based fallback if LLM parsing fails.
    """
    normalized = normalize_common_typos(user_message)

    item_id = extract_item_id(user_message)
    image_path = extract_image_path(user_message)

    if item_id:
        return {
            "raw_query": user_message,
            "corrected_query": normalized,
            "intent": "wardrobe_item_pairing",
            "tool_action": "style_wardrobe_item",
            "occasion": None,
            "vibe": None,
            "color_preference": None,
            "item_id": item_id,
            "image_path": None,
            "is_followup": False,
            "needs_followup": False,
            "followup_question": None,
            "refinement_request": None,
        }

    if image_path:
        if any(word in normalized for word in ["similar", "same", "like this"]):
            intent = "outside_image_similar"
            tool_action = "find_similar_for_outside_image"
        else:
            intent = "outside_image_pairing"
            tool_action = "style_outside_image"

        return {
            "raw_query": user_message,
            "corrected_query": normalized,
            "intent": intent,
            "tool_action": tool_action,
            "occasion": None,
            "vibe": None,
            "color_preference": None,
            "item_id": None,
            "image_path": image_path,
            "is_followup": False,
            "needs_followup": False,
            "followup_question": None,
            "refinement_request": None,
        }

    if any(
        word in normalized
        for word in [
            "party",
            "office",
            "date",
            "wedding",
            "outing",
            "travel",
            "casual",
        ]
    ):
        return {
            "raw_query": user_message,
            "corrected_query": normalized,
            "intent": "text_outfit",
            "tool_action": "recommend_outfit_from_text",
            "occasion": normalized,
            "vibe": None,
            "color_preference": None,
            "item_id": None,
            "image_path": None,
            "is_followup": False,
            "needs_followup": False,
            "followup_question": None,
            "refinement_request": None,
        }

    return {
        "raw_query": user_message,
        "corrected_query": normalized,
        "intent": "unclear",
        "tool_action": "ask_followup",
        "occasion": None,
        "vibe": None,
        "color_preference": None,
        "item_id": None,
        "image_path": None,
        "is_followup": False,
        "needs_followup": True,
        "followup_question": (
            "Sure, I can help. Tell me the occasion, like party, office, date, "
            "casual outing, or share an item_id/image you want me to style."
        ),
        "refinement_request": None,
    }


def understand_user_query(
    user_message: str,
    chat_context: dict,
    has_previous_recommendation: bool,
) -> dict:
    """
    Main query understanding layer.
    Converts messy user message into clean structured system intent.
    """
    followup_result = infer_followup_without_llm(
        user_message=user_message,
        chat_context=chat_context,
    )

    if followup_result:
        return followup_result

    normalized_message = normalize_common_typos(user_message)
    detected_item_id = extract_item_id(user_message)
    detected_image_path = extract_image_path(user_message)

    prompt = f"""
You are a query understanding layer for a fashion stylist assistant.

Your job:
Convert the user's raw message into clean structured JSON that the backend system can use.

Current chat context:
{safe_json(chat_context)}

Has previous recommendation:
{has_previous_recommendation}

Raw user message:
"{user_message}"

Typo-normalized message:
"{normalized_message}"

Detected item_id by regex:
{detected_item_id}

Detected image_path by regex:
{detected_image_path}

Possible intent values:
- text_outfit
- wardrobe_item_pairing
- outside_image_pairing
- outside_image_similar
- refine_previous
- unclear

Possible tool_action values:
- recommend_outfit_from_text
- style_wardrobe_item
- style_outside_image
- find_similar_for_outside_image
- refine_previous_recommendation
- ask_followup

Rules:
- Correct spelling mistakes and create a clean corrected_query.
- If user asks what to wear without item/image, intent is text_outfit.
- If user says party, office, date, casual outing, wedding, or travel, intent is text_outfit.
- If user mentions item_id like item_003 and asks pair/style/wear, intent is wardrobe_item_pairing.
- If user mentions an image path and asks pair/style/wear/outfit, intent is outside_image_pairing.
- If user mentions an image path and asks similar/same/like this, intent is outside_image_similar.
- If user says make it casual/formal/bold/classy/change/another option and previous recommendation exists, intent is refine_previous.
- If current context has pending_action and the user gives a short answer, treat it as follow-up, not unclear.
- Be tolerant of typos and broken English.
- If occasion is missing for an outfit request, needs_followup should be true.
- If text_outfit has occasion but vibe is missing, ask a vibe follow-up.
- Do not ask follow-up if enough info is available.
- For a fresh request, do not copy old vibe/item/image from context unless the user is clearly referring to previous context.
- For "what we have now", use previous image_path when available and intent should usually be outside_image_similar.

Return only valid JSON in this exact structure:

{{
  "raw_query": "{user_message}",
  "corrected_query": "",
  "intent": "",
  "tool_action": "",
  "occasion": null,
  "vibe": null,
  "color_preference": null,
  "item_id": null,
  "image_path": null,
  "is_followup": false,
  "needs_followup": false,
  "followup_question": null,
  "refinement_request": null
}}
"""

    try:
        response = client.responses.create(
            model=QUERY_UNDERSTANDING_MODEL,
            input=prompt,
        )

        parsed = extract_json_from_text(response.output_text.strip())

    except Exception:
        parsed = fallback_understanding(
            user_message=user_message,
            chat_context=chat_context,
        )

    if detected_item_id:
        parsed["item_id"] = detected_item_id
        parsed["intent"] = "wardrobe_item_pairing"
        parsed["tool_action"] = "style_wardrobe_item"

    if detected_image_path:
        parsed["image_path"] = detected_image_path

        if any(word in normalized_message for word in ["similar", "same", "like this"]):
            parsed["intent"] = "outside_image_similar"
            parsed["tool_action"] = "find_similar_for_outside_image"
        else:
            parsed["intent"] = "outside_image_pairing"
            parsed["tool_action"] = "style_outside_image"

    if (
        parsed.get("intent") == "outside_image_similar"
        and not parsed.get("image_path")
        and chat_context.get("image_path")
    ):
        parsed["image_path"] = chat_context.get("image_path")

    if not parsed.get("corrected_query"):
        parsed["corrected_query"] = normalized_message

    return parsed