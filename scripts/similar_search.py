from src.engine.similar_search import (
    print_similar_results,
    similar_search_by_item_id,
    similar_search_by_new_image,
)


def main():
    print("Similar Search")
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

        results = similar_search_by_item_id(
            item_id=item_id,
            top_k=5,
            hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
        )

        print_similar_results(
            results,
            title=f"SIMILAR ITEMS FOR {item_id}",
        )

    elif choice == "2":
        image_path = input(
            "Enter new image path, example data/new_query_images/shopping_dress.jpg: "
        ).strip()

        if not image_path:
            print("Image path cannot be empty.")
            return

        results = similar_search_by_new_image(
            image_path=image_path,
            top_k=5,
            hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
        )

        print_similar_results(
            results,
            title="SIMILAR ITEMS FOR NEW IMAGE",
        )

    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()