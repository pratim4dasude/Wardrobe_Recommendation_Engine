from src.db.pgvector_store import (
    fetch_item_by_id,
    get_item_count,
    hybrid_search_text_and_visual,
    search_by_text_embedding,
)
from src.engine.embeddings import create_text_embedding


def print_results(title: str, results: list[dict]) -> None:
    """
    Print pgvector search results.
    """
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No results found.")
        return

    for rank, item in enumerate(results, start=1):
        print(f"\nRank #{rank}")
        print(f"Item ID: {item.get('item_id')}")
        print(f"Filename: {item.get('filename')}")
        print(f"Item type: {item.get('item_type')}")
        print(f"Text score: {item.get('text_score')}")
        print(f"Visual score: {item.get('visual_score')}")
        print(f"Hybrid score: {item.get('hybrid_pgvector_score')}")
        print(f"Similarity score: {item.get('similarity_score')}")
        print(f"Caption: {item.get('caption')}")


def run_text_search() -> None:
    """
    Run text-only pgvector search.
    """
    query = "smart clean office outfit with white shirt and navy trousers"

    print("\nCreating text embedding for query...")
    query_embedding = create_text_embedding(query)

    print(f"Text query vector size: {len(query_embedding)}")

    results = search_by_text_embedding(
        query_embedding=query_embedding,
        top_k=5,
    )

    print_results("PGVECTOR TEXT SEARCH RESULTS", results)


def run_hybrid_search_from_existing_item() -> None:
    """
    Run hybrid text + visual search using existing wardrobe item.
    """
    item_id = "item_003"

    print(f"\nFetching source item from pgvector DB: {item_id}")
    source_item = fetch_item_by_id(item_id)

    if not source_item:
        raise ValueError(f"Item not found in pgvector DB: {item_id}")

    print(f"Source item caption: {source_item.get('caption')}")
    print(f"Source text vector size: {len(source_item.get('text_embedding', []))}")
    print(f"Source visual vector size: {len(source_item.get('visual_embedding', []))}")

    results = hybrid_search_text_and_visual(
        text_embedding=source_item["text_embedding"],
        visual_embedding=source_item["visual_embedding"],
        top_k=5,
        exclude_item_id=item_id,
    )

    print_results("PGVECTOR HYBRID TEXT + VISUAL SEARCH RESULTS", results)


def main() -> None:
    count = get_item_count()
    print("=" * 80)
    print("PGVECTOR DB CHECK")
    print("=" * 80)
    print(f"Total items in pgvector DB: {count}")

    run_text_search()
    run_hybrid_search_from_existing_item()


if __name__ == "__main__":
    main()