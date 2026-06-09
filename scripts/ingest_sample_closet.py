from src.engine.captioner import process_inventory


def main():
    process_inventory(
        inventory_path="data/wardrobe_inventory.json",
        output_path="data/wardrobe_metadata.json",
    )


if __name__ == "__main__":
    main()