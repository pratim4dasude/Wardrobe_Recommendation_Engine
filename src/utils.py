from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def get_project_root() -> Path:
    """
    Return absolute project root path.

    Example:
    C:/Users/KIIT/PycharmProjects/Wardrobe_Recommendation_Engine
    """
    return PROJECT_ROOT


def resolve_project_path(path: str | Path) -> Path:
    """
    Convert any relative path into an absolute path from project root.

    Example:
    data/uploaded_images
    becomes:
    C:/Users/KIIT/PycharmProjects/Wardrobe_Recommendation_Engine/data/uploaded_images
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def ensure_dir(path: str | Path) -> Path:
    """
    Create folder from project root if it does not exist.
    """
    folder_path = resolve_project_path(path)
    folder_path.mkdir(parents=True, exist_ok=True)
    return folder_path


def is_supported_image(file_path: str | Path) -> bool:
    """
    Check if file is a supported image.
    """
    return Path(file_path).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS


def list_uploaded_images(upload_dir: str | Path = "data/uploaded_images") -> list[Path]:
    """
    List all images from uploaded_images folder.
    Always reads from project root.
    """
    upload_path = ensure_dir(upload_dir)

    image_files = [
        file_path
        for file_path in upload_path.iterdir()
        if file_path.is_file() and is_supported_image(file_path)
    ]

    return sorted(image_files)