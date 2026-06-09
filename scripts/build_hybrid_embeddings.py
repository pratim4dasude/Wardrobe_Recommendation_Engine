from src.engine.visual_embeddings import generate_visual_and_hybrid_embeddings


def main():
    generate_visual_and_hybrid_embeddings(
        text_embeddings_path="data/wardrobe_embeddings.json",
        output_path="data/wardrobe_hybrid_embeddings.json",
    )


if __name__ == "__main__":
    main()