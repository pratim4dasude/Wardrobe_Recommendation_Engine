import json

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI()

RERANK_MODEL = "gpt-4.1-mini"


def build_candidate_context(results: list[dict]) -> str:
    """
    Convert retrieved wardrobe items into clean LLM-readable context.
    We do not send embeddings to LLM, only useful metadata and scores.
    """
    context_items = []

    for index, item in enumerate(results, start=1):
        context_items.append(
            {
                "candidate_rank": index,
                "item_id": item.get("item_id"),
                "filename": item.get("filename"),
                "image_path": item.get("image_path"),
                "retrieval_scores": {
                    "final_score": item.get("final_score"),
                    "text_score": item.get("text_score"),
                    "visual_score": item.get("visual_score"),
                    "similarity_score": item.get("similarity_score"),
                },
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
    Safely parse JSON from LLM response.
    Handles pure JSON or accidental surrounding text.
    """
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1 and end > start:
        json_text = text[start : end + 1]
        return json.loads(json_text)

    raise ValueError("No valid JSON found in LLM response.")


def llm_rerank_search_results(
    user_query: str,
    retrieved_results: list[dict],
    top_k: int = 5,
) -> dict:
    """
    RAG-style reranking.

    Input:
    - user query
    - retrieved wardrobe candidates

    Output:
    - LLM-ranked fashion recommendations with reasons.
    """
    if not retrieved_results:
        return {
            "user_query": user_query,
            "recommended_items": [],
            "summary": "No retrieved items were provided.",
        }

    candidate_context = build_candidate_context(retrieved_results)

    prompt = f"""
You are a fashion recommendation reranker.

The user asked:
"{user_query}"

You are given wardrobe items retrieved by a hybrid search engine.
The retrieval engine already used text embeddings and visual embeddings.
Your job is to rerank these items using fashion reasoning.

Important:
- Prefer exact intent match over only high retrieval score.
- Consider category, color, style, fit, occasion, and caption.
- If an item partially matches, explain why.
- Do not invent new items.
- Only choose from the provided candidates.
- Return only valid JSON.
- Do not include markdown.

Retrieved wardrobe candidates:
{candidate_context}

Return JSON in this exact structure:

{{
  "user_query": "{user_query}",
  "summary": "",
  "recommended_items": [
    {{
      "rank": 1,
      "item_id": "",
      "filename": "",
      "image_path": "",
      "match_type": "exact | strong | partial",
      "reason": "",
      "styling_tip": "",
      "why_not_lower_rank": ""
    }}
  ]
}}

Rules:
- Return only top {top_k} items.
- rank must start from 1.
- match_type must be one of: exact, strong, partial.
- summary should explain the overall recommendation in 1-2 lines.
"""

    response = client.responses.create(
        model=RERANK_MODEL,
        input=prompt,
    )

    raw_text = response.output_text.strip()

    try:
        parsed_response = extract_json_from_text(raw_text)
    except Exception as error:
        return {
            "user_query": user_query,
            "recommended_items": [],
            "summary": "LLM reranking failed.",
            "error": str(error),
            "raw_response": raw_text,
        }

    return parsed_response


def print_llm_reranked_results(reranked_response: dict) -> None:
    """
    Print LLM reranked results in terminal.
    """
    print("\n" + "=" * 80)
    print("LLM RERANKED FASHION RECOMMENDATIONS")
    print("=" * 80)

    print(f"\nUser query: {reranked_response.get('user_query', '')}")
    print(f"Summary: {reranked_response.get('summary', '')}")

    recommended_items = reranked_response.get("recommended_items", [])

    if not recommended_items:
        print("\nNo recommended items found.")
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
        print(f"Styling tip: {item.get('styling_tip')}")
        print(f"Why not lower rank: {item.get('why_not_lower_rank')}")