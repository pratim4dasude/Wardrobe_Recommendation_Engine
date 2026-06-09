import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()

SIMILAR_RERANK_MODEL = "gpt-4.1-mini"


def build_source_item_context(source_item: dict | None, source_description: str = "") -> dict:
    """
    Build source item context for LLM.
    If source item exists, use wardrobe metadata.
    If source is a new image, use simple source description.
    """
    if source_item:
        return {
            "item_id": source_item.get("item_id"),
            "filename": source_item.get("filename"),
            "image_path": source_item.get("image_path"),
            "caption": source_item.get("caption", ""),
            "category": source_item.get("category", []),
            "color": source_item.get("color", []),
            "style": source_item.get("style", []),
            "search_text": source_item.get("search_text", ""),
        }

    return {
        "source_type": "new_uploaded_image",
        "description": source_description or "New uploaded image used for visual similarity search.",
    }


def build_candidate_context(results: list[dict]) -> str:
    """
    Convert similar search candidates into clean LLM-readable JSON.
    """
    context_items = []

    for index, item in enumerate(results, start=1):
        context_items.append(
            {
                "candidate_rank": index,
                "item_id": item.get("item_id"),
                "filename": item.get("filename"),
                "image_path": item.get("image_path"),
                "similarity_score": item.get("similarity_score"),
                "caption": item.get("caption", ""),
                "category": item.get("category", []),
                "color": item.get("color", []),
                "style": item.get("style", []),
                "search_text": item.get("search_text", ""),
            }
        )

    return json.dumps(context_items, indent=2, ensure_ascii=False)


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


def llm_rerank_similar_results(
    retrieved_results: list[dict],
    source_item: dict | None = None,
    source_description: str = "",
    top_k: int = 5,
) -> dict:
    """
    RAG-style LLM reranking for similar item search.

    It takes visually retrieved candidates and asks LLM to judge
    actual fashion similarity.
    """
    if not retrieved_results:
        return {
            "summary": "No similar candidates were retrieved.",
            "recommended_items": [],
        }

    source_context = build_source_item_context(
        source_item=source_item,
        source_description=source_description,
    )

    candidate_context = build_candidate_context(retrieved_results)

    prompt = f"""
You are a fashion similarity reranker.

Your task:
Given a source clothing item and visually retrieved wardrobe candidates,
rerank the candidates based on true fashion similarity.

Consider:
- clothing category
- garment type
- color family
- pattern
- fit
- fabric guess
- style
- occasion
- whether the item is actually similar or only weakly related

Important:
- Do not invent new items.
- Only choose from the provided candidates.
- Prefer same clothing category when possible.
- Prefer similar shape, structure, color/pattern, and use case.
- Return only valid JSON.
- Do not include markdown.

Source item:
{json.dumps(source_context, indent=2, ensure_ascii=False)}

Retrieved similar candidates:
{candidate_context}

Return JSON in this exact structure:

{{
  "summary": "",
  "recommended_items": [
    {{
      "rank": 1,
      "item_id": "",
      "filename": "",
      "image_path": "",
      "match_type": "very_similar | similar | weak_similar",
      "reason": "",
      "similarity_explanation": "",
      "difference_from_source": ""
    }}
  ]
}}

Rules:
- Return only top {top_k} items.
- rank must start from 1.
- match_type must be one of: very_similar, similar, weak_similar.
- summary should explain the overall similarity result in 1-2 lines.
"""

    response = client.responses.create(
        model=SIMILAR_RERANK_MODEL,
        input=prompt,
    )

    raw_text = response.output_text.strip()

    try:
        return extract_json_from_text(raw_text)
    except Exception as error:
        return {
            "summary": "LLM similar reranking failed.",
            "recommended_items": [],
            "error": str(error),
            "raw_response": raw_text,
        }


def print_llm_similar_results(reranked_response: dict) -> None:
    """
    Print LLM reranked similar search results.
    """
    print("\n" + "=" * 80)
    print("LLM RERANKED SIMILAR ITEMS")
    print("=" * 80)

    print(f"\nSummary: {reranked_response.get('summary', '')}")

    recommended_items = reranked_response.get("recommended_items", [])

    if not recommended_items:
        print("\nNo recommended similar items found.")
        if reranked_response.get("error"):
            print(f"Error: {reranked_response.get('error')}")
        return

    for item in recommended_items:
        print("\n" + "-" * 80)
        print(f"Rank #{item.get('rank')}")
        print(f"Item ID: {item.get('item_id')}")
        print(f"Filename: {item.get('filename')}")
        print(f"Image path: {item.get('image_path')}")
        print(f"Match type: {item.get('match_type')}")
        print(f"Reason: {item.get('reason')}")
        print(f"Similarity explanation: {item.get('similarity_explanation')}")
        print(f"Difference from source: {item.get('difference_from_source')}")