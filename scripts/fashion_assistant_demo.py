from src.engine.fashion_assistant import (
    find_similar_for_new_image,
    print_any_outfit_recommendation,
    recommend_outfit_from_new_image,
    recommend_outfit_from_text_query,
    recommend_outfit_from_wardrobe_item,
)
from src.engine.llm_similar_reranker import print_llm_similar_results


def main():
    print("Fashion Assistant Demo")
    print("=" * 80)
    print("Choose request type:")
    print("1. Text-only outfit request")
    print("2. Pair outfit with wardrobe item")
    print("3. Pair outfit with outside image")
    print("4. Find similar wardrobe items for outside image")
    print("=" * 80)

    choice = input("Enter choice 1, 2, 3, or 4: ").strip()

    if choice == "1":
        user_query = input(
            "Enter request, example: I need to go to a party tonight: "
        ).strip()

        if not user_query:
            print("Request cannot be empty.")
            return

        recommendation = recommend_outfit_from_text_query(
            user_query=user_query,
            hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
        )

        print_any_outfit_recommendation(recommendation)

    elif choice == "2":
        item_id = input("Enter wardrobe item_id, example item_003: ").strip()

        if not item_id:
            print("item_id cannot be empty.")
            return

        user_query = input(
            "Enter request, example: smart casual outing / party night: "
        ).strip()

        if not user_query:
            user_query = "casual everyday outfit"

        recommendation = recommend_outfit_from_wardrobe_item(
            item_id=item_id,
            user_query=user_query,
            hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
        )

        print_any_outfit_recommendation(recommendation)

    elif choice == "3":
        image_path = input(
            "Enter outside image path, example C:\\Users\\KIIT\\...\\11.jpg: "
        ).strip()

        if not image_path:
            print("Image path cannot be empty.")
            return

        user_query = input(
            "Enter request, example: what can I pair with this for a party: "
        ).strip()

        if not user_query:
            user_query = "suggest a complete outfit using this item"

        recommendation = recommend_outfit_from_new_image(
            image_path=image_path,
            user_query=user_query,
            hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
        )

        print_any_outfit_recommendation(recommendation)

    elif choice == "4":
        image_path = input(
            "Enter outside image path, example C:\\Users\\KIIT\\...\\11.jpg: "
        ).strip()

        if not image_path:
            print("Image path cannot be empty.")
            return

        reranked_response = find_similar_for_new_image(
            image_path=image_path,
            hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
        )

        print_llm_similar_results(reranked_response)

    else:
        print("Invalid choice. Please enter 1, 2, 3, or 4.")


if __name__ == "__main__":
    main()