from src.engine.similar_search import (
    find_item_by_id,
    load_hybrid_items,
    similar_search_by_item_id,
    similar_search_by_new_image,
)
from src.engine.llm_similar_reranker import (
    llm_rerank_similar_results,
    print_llm_similar_results,
)


def main():
    print("Similar Search with LLM Reranker")
    print("=" * 80)
    print("Choose input type:")
    print("1. Existing wardrobe item_id")
    print("2. New image path")
    print("=" * 80)

    choice = input("Enter choice 1 or 2: ").strip()

    if choice == "1":
        item_id = input("Enter wardrobe item_id, example item_001: ").strip()

        if not item_id:
            print("item_id cannot be empty.")
            return

        print("\nStep 1: Running visual similar search...")

        retrieved_results = similar_search_by_item_id(
            item_id=item_id,
            top_k=10,
            hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
        )

        all_items = load_hybrid_items("data/wardrobe_hybrid_embeddings.json")
        source_item = find_item_by_id(all_items, item_id)

        print(f"Retrieved {len(retrieved_results)} similar candidates.")

        print("\nStep 2: Sending source item + candidates to LLM reranker...")

        reranked_response = llm_rerank_similar_results(
            source_item=source_item,
            retrieved_results=retrieved_results,
            top_k=5,
        )

        print_llm_similar_results(reranked_response)

    elif choice == "2":
        image_path = input(
            "Enter new image path, example data/new_query_images/shopping_dress.jpg: "
        ).strip()

        if not image_path:
            print("Image path cannot be empty.")
            return

        print("\nStep 1: Processing new image like a temporary wardrobe item...")
        print("This creates caption + text embedding + visual embedding.")

        query_item, retrieved_results = similar_search_by_new_image(
            image_path=image_path,
            top_k=10,
            hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
        )

        print(f"\nRetrieved {len(retrieved_results)} similar candidates.")

        print("\nStep 2: Sending temporary query item + candidates to LLM reranker...")

        reranked_response = llm_rerank_similar_results(
            source_item=query_item,
            retrieved_results=retrieved_results,
            top_k=5,
        )

        print_llm_similar_results(reranked_response)

    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()