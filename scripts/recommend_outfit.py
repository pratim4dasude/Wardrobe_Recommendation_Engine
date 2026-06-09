from src.engine.outfit_recommender import (
    print_outfit_recommendations,
    recommend_outfits_for_item,
)


def main():
    print("Outfit Compatibility Recommender")
    print("=" * 80)

    item_id = input("Enter source wardrobe item_id, example item_001: ").strip()

    if not item_id:
        print("item_id cannot be empty.")
        return

    occasion = input(
        "Enter occasion/style intent, example casual summer, office, party: "
    ).strip()

    if not occasion:
        occasion = "casual everyday"

    recommendation = recommend_outfits_for_item(
        item_id=item_id,
        occasion=occasion,
        hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
    )

    print_outfit_recommendations(recommendation)


if __name__ == "__main__":
    main()