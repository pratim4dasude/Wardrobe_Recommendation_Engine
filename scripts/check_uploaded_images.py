import json

from src.utils import get_project_root, is_supported_image, resolve_project_path
from PIL import Image, ImageDraw

UPLOAD_DIR = "data/uploaded_images"
INVENTORY_PATH = "data/wardrobe_inventory.json"


def get_uploaded_image_files(upload_dir: str = UPLOAD_DIR):
    upload_path = resolve_project_path(upload_dir)

    if not upload_path.exists():
        upload_path.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        [
            file_path
            for file_path in upload_path.iterdir()
            if file_path.is_file() and is_supported_image(file_path)
        ]
    )

    return image_files


def rename_uploaded_images(upload_dir: str = UPLOAD_DIR) -> None:
    image_files = get_uploaded_image_files(upload_dir)

    if not image_files:
        print("No image files found to rename.")
        return

    temp_files = []
    upload_path = resolve_project_path(upload_dir)

    for index, image_path in enumerate(image_files, start=1):
        temp_path = upload_path / f"temp_rename_{index}{image_path.suffix.lower()}"
        image_path.rename(temp_path)
        temp_files.append(temp_path)

    for index, temp_path in enumerate(temp_files, start=1):
        final_name = f"item_{index:03d}{temp_path.suffix.lower()}"
        final_path = upload_path / final_name
        temp_path.rename(final_path)

    print(f"Renamed {len(temp_files)} image(s).")


def create_wardrobe_inventory() -> None:
    image_files = get_uploaded_image_files(UPLOAD_DIR)

    if not image_files:
        print("No images found. Inventory not created.")
        return

    inventory = []

    for image_path in image_files:
        item = {
            "item_id": image_path.stem,
            "filename": image_path.name,
            "image_path": f"{UPLOAD_DIR}/{image_path.name}",
            "status": "pending_metadata",
        }
        inventory.append(item)

    output_path = resolve_project_path(INVENTORY_PATH)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(inventory, file, indent=2)

    print(f"Created wardrobe inventory with {len(inventory)} item(s).")
    print(f"Inventory saved at: {output_path}")


def check_uploaded_images() -> None:
    image_files = get_uploaded_image_files(UPLOAD_DIR)

    if not image_files:
        print("No images found in data/uploaded_images/")
        return

    print(f"Found {len(image_files)} uploaded image(s):")

    for image in image_files:
        print(f"- {image.name}")


def create_wardrobe_preview_sheet() -> None:
    image_files = get_uploaded_image_files(UPLOAD_DIR)

    if not image_files:
        print("No images found. Preview sheet not created.")
        return

    thumbnail_size = (160, 160)
    columns = 6
    padding = 20
    label_height = 30

    rows = (len(image_files) + columns - 1) // columns

    sheet_width = columns * (thumbnail_size[0] + padding) + padding
    sheet_height = rows * (thumbnail_size[1] + label_height + padding) + padding

    preview_sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    draw = ImageDraw.Draw(preview_sheet)

    for index, image_path in enumerate(image_files):
        row = index // columns
        col = index % columns

        x = padding + col * (thumbnail_size[0] + padding)
        y = padding + row * (thumbnail_size[1] + label_height + padding)

        try:
            image = Image.open(image_path).convert("RGB")
            image.thumbnail(thumbnail_size)

            preview_sheet.paste(image, (x, y))

            label = image_path.stem
            draw.text((x, y + thumbnail_size[1] + 5), label, fill="black")

        except Exception as error:
            draw.text((x, y), f"Error: {image_path.name}", fill="black")
            print(f"Could not process {image_path.name}: {error}")

    output_dir = resolve_project_path("outputs/sample_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "wardrobe_preview.jpg"
    preview_sheet.save(output_path)

    print(f"Created wardrobe preview sheet at: {output_path}")


def main():
    print(f"Project root: {get_project_root()}")
    print(f"Upload folder: {resolve_project_path(UPLOAD_DIR)}")
    print()

    rename_uploaded_images()
    print()

    check_uploaded_images()
    print()

    create_wardrobe_inventory()
    print()
    create_wardrobe_preview_sheet()

if __name__ == "__main__":
    main()