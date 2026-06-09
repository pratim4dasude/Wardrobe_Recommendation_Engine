from src.engine.hybrid_retrieval import hybrid_search
from src.engine.llm_reranker import (
    llm_rerank_search_results,
    print_llm_reranked_results,
)


def main():
    query = input("Enter wardrobe search query: ").strip()

    if not query:
        print("Query cannot be empty.")
        return

    print("\nStep 1: Running hybrid retrieval...")

    retrieved_results = hybrid_search(
        query=query,
        top_k=10,
        hybrid_embeddings_path="data/wardrobe_hybrid_embeddings.json",
    )

    print(f"\nRetrieved {len(retrieved_results)} candidate items.")

    print("\nStep 2: Sending retrieved items to LLM reranker...")

    reranked_response = llm_rerank_search_results(
        user_query=query,
        retrieved_results=retrieved_results,
        top_k=5,
    )

    print_llm_reranked_results(reranked_response)


if __name__ == "__main__":
    main()