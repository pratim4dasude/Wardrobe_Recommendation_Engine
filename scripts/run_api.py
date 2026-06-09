from src.engine.embeddings import generate_embeddings_from_metadata


def main():
    generate_embeddings_from_metadata(
        metadata_path="data/wardrobe_metadata.json",
        output_path="data/wardrobe_embeddings.json",
    )


if __name__ == "__main__":
    main()