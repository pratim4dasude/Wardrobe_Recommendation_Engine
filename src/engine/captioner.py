import base64
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.utils import resolve_project_path


load_dotenv()

client = OpenAI()

MODEL_NAME = "gpt-4.1-mini"

# Delay between caption call and category call
CAPTION_SLEEP_SECONDS = 1.5

# Delay after each image item is completed
ITEM_SLEEP_SECONDS = 1.0


def encode_image(image_path: str | Path) -> str:
    """
    Convert image into base64 string for OpenAI vision input.
    """
    image_path = resolve_project_path(image_path)

    with image_path.open("rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def get_mime_type(image_path: str | Path) -> str:
    """
    Detect image MIME type.
    """
    suffix = Path(image_path).suffix.lower()

    if suffix in [".jpg", ".jpeg"]:
        return "image/jpeg"

    if suffix == ".png":
        return "image/png"

    if suffix == ".webp":
        return "image/webp"

    raise ValueError(f"Unsupported image type: {suffix}")


def caption_main_clothing(image_path: str | Path) -> str:
    """
    First OpenAI call:
    Image -> one clean fashion caption.
    """
    image_path = resolve_project_path(image_path)
    image_base64 = encode_image(image_path)
    mime_type = get_mime_type(image_path)

    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            "Caption the main clothing item in this image. "
                            "Focus only on the visible clothing item, not the person or background. "
                            "Mention clothing type, color, pattern if any, style, fit, fabric guess, "
                            "and where it can be worn. "
                            "Return one clear sentence only."
                        ),
                    },
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{image_base64}",
                    },
                ],
            }
        ],
    )

    return response.output_text.strip()


def extract_json_from_text(text: str) -> dict:
    """
    Parse JSON from model response.

    This handles both:
    1. Pure JSON
    2. Text containing JSON accidentally
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

    raise json.JSONDecodeError("No valid JSON found", text, 0)


def normalize_metadata(metadata: dict, fallback_caption: str) -> dict:
    """
    Keep metadata structure simple and consistent.
    """
    caption = metadata.get("caption") or fallback_caption
    category = metadata.get("category")
    color = metadata.get("color")
    style = metadata.get("style")
    search_text = metadata.get("search_text") or fallback_caption

    return {
        "caption": caption,
        "category": category if isinstance(category, list) else [],
        "color": color if isinstance(color, list) else [],
        "style": style if isinstance(style, list) else [],
        "search_text": search_text,
    }


def convert_caption_to_category_json(caption: str) -> dict:
    """
    Second OpenAI call:
    Caption -> simple fashion retrieval JSON.
    """
    prompt = f"""
Convert this clothing caption into simple fashion retrieval JSON.

Caption:
{caption}

Return only valid JSON in this exact structure:

{{
  "caption": "",
  "category": [],
  "color": [],
  "style": [],
  "search_text": ""
}}

Rules:
- caption should be a clean improved version of the original caption.
- category should include 2 to 4 useful labels.
- category examples: tank top, sleeveless top, shirt, t-shirt, dress, jeans, trousers, jacket, hoodie, shoes, sneakers, accessory, upperwear, bottomwear, footwear.
- color should include visible clothing colors only.
- style should include 2 to 5 useful tags.
- style examples: casual, formal, party, summer, winter, minimal, streetwear, office, sporty, ethnic, daily wear.
- search_text should be one rich sentence useful for semantic search and future embeddings.
- Do not include markdown.
- Do not include explanation.
- Return JSON only.
"""

    response = client.responses.create(
        model=MODEL_NAME,
        input=prompt,
    )

    raw_text = response.output_text.strip()

    try:
        metadata = extract_json_from_text(raw_text)
    except json.JSONDecodeError:
        metadata = {
            "caption": caption,
            "category": [],
            "color": [],
            "style": [],
            "search_text": caption,
        }

    return normalize_metadata(metadata, fallback_caption=caption)


def process_single_item(item: dict) -> dict:
    """
    Process one wardrobe item:
    image -> caption -> simple category JSON.
    """
    image_path = item["image_path"]

    print("Step 1: Generating caption...")
    caption = caption_main_clothing(image_path)
    print(f"Caption: {caption}")

    time.sleep(CAPTION_SLEEP_SECONDS)

    print("Step 2: Converting caption to category JSON...")
    metadata = convert_caption_to_category_json(caption)

    final_item = {
        "item_id": item["item_id"],
        "filename": item["filename"],
        "image_path": item["image_path"],
        "status": "metadata_generated",
        "caption": metadata["caption"],
        "category": metadata["category"],
        "color": metadata["color"],
        "style": metadata["style"],
        "search_text": metadata["search_text"],
        "raw_caption": caption,
    }

    return final_item


def save_results(output_file: Path, results: list[dict]) -> None:
    """
    Save metadata results to JSON.
    """
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def process_inventory(
    inventory_path: str = "data/wardrobe_inventory.json",
    output_path: str = "data/wardrobe_metadata.json",
) -> None:
    """
    Reads wardrobe inventory and creates wardrobe metadata JSON.

    It prints every generated item in terminal immediately.
    It also saves after every item, so progress is not lost if script stops.
    """
    inventory_file = resolve_project_path(inventory_path)
    output_file = resolve_project_path(output_path)

    if not inventory_file.exists():
        raise FileNotFoundError(f"Inventory file not found: {inventory_file}")

    with inventory_file.open("r", encoding="utf-8") as file:
        inventory = json.load(file)

    results = []

    for index, item in enumerate(inventory, start=1):
        print("\n" + "=" * 80)
        print(f"[{index}/{len(inventory)}] Processing {item['filename']}")
        print("=" * 80)

        try:
            final_item = process_single_item(item)
            results.append(final_item)

            print("\nGenerated metadata:")
            print(json.dumps(final_item, indent=2, ensure_ascii=False))

            save_results(output_file, results)

            print(f"\nSaved metadata for {item['filename']}")
            print(f"Progress: {index}/{len(inventory)} completed")

        except Exception as error:
            print(f"\nFailed to process {item['filename']}: {error}")

            failed_item = {
                "item_id": item.get("item_id", "unknown"),
                "filename": item.get("filename", "unknown"),
                "image_path": item.get("image_path", ""),
                "status": "failed",
                "caption": "",
                "category": [],
                "color": [],
                "style": [],
                "search_text": "",
                "raw_caption": "",
                "error": str(error),
            }

            results.append(failed_item)

            print("\nFailed item:")
            print(json.dumps(failed_item, indent=2, ensure_ascii=False))

            save_results(output_file, results)

        time.sleep(ITEM_SLEEP_SECONDS)

    print("\n" + "=" * 80)
    print(f"Metadata generated for {len(results)} item(s).")
    print(f"Saved at: {output_file}")
    print("=" * 80)