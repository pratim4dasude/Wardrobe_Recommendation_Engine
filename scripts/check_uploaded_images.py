from src.utils import get_project_root, list_uploaded_images


def main():
    print(f"Project root: {get_project_root()}")

    images = list_uploaded_images("data/uploaded_images")

    if not images:
        print("No images found in data/uploaded_images/")
        return

    print(f"Found {len(images)} uploaded image(s):")

    for image in images:
        print(f"- {image}")


if __name__ == "__main__":
    main()