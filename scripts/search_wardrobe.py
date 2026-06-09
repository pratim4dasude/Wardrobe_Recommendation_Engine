from src.engine.hybrid_retrieval import hybrid_search, print_search_results


def main():
    query = input("Enter wardrobe search query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    results = hybrid_search(
        query=query,
        top_k=5,
        hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
    )

    print_search_results(results)


if __name__ == "__main__":
    main()