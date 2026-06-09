import json
from pathlib import Path

from src.db.pgvector_store import (
    create_pgvector_schema,
    get_item_count,
    upsert_wardrobe_item,
)
from src.utils import resolve_project_path


HYBRID_JSON_PATH = "data/wardrobe_hybrid_embeddings.json"


def main() -> None:
    json_path = resolve_project_path(HYBRID_JSON_PATH)

    if not Path(json_path).exists():
        raise FileNotFoundError(f"Could not find {json_path}")

    print("Creating pgvector schema...")
    create_pgvector_schema()

    print(f"Loading wardrobe items from {json_path}...")

    with open(json_path, "r", encoding="utf-8") as file:
        items = json.load(file)

    print(f"Found {len(items)} items in JSON.")

    inserted = 0
    skipped = 0

    for item in items:
        success = upsert_wardrobe_item(item)

        if success:
            inserted += 1
        else:
            skipped += 1

    db_count = get_item_count()

    print("\n" + "=" * 80)
    print("PGVECTOR MIGRATION COMPLETE")
    print("=" * 80)
    print(f"Inserted/updated: {inserted}")
    print(f"Skipped: {skipped}")
    print(f"Total items in DB: {db_count}")


if __name__ == "__main__":
    main()