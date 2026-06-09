import os
import runpy
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.db.pgvector_store import get_item_count
from src.db.postgres_chat_memory import create_chat_memory_schema
from src.engine.fashion_chat_assistant import FashionChatAssistant
from src.utils import resolve_project_path


load_dotenv()


PROJECT_ROOT = Path(__file__).resolve().parent

UPLOADED_IMAGES_DIR = resolve_project_path("data/uploaded_images")

WARDROBE_INVENTORY_PATH = resolve_project_path("data/wardrobe_inventory.json")
WARDROBE_METADATA_PATH = resolve_project_path("data/wardrobe_metadata.json")
WARDROBE_TEXT_EMBEDDINGS_PATH = resolve_project_path("data/wardrobe_embeddings.json")
WARDROBE_HYBRID_EMBEDDINGS_PATH = resolve_project_path(
    "data/wardrobe_hybrid_embeddings.json"
)


def print_header(title: str) -> None:
    """
    Print clean section header.
    """
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def print_step(message: str) -> None:
    """
    Print pipeline step.
    """
    print(f"\n[MAIN] {message}")


def run_script_module(module_name: str) -> None:
    """
    Run an existing script as a Python module.

    This keeps main.py clean and lets us reuse the scripts already built
    in the project.
    """
    print_step(f"Running module: {module_name}")
    runpy.run_module(module_name, run_name="__main__")


def file_exists(path: Path) -> bool:
    """
    Check whether a file exists and is not empty.
    """
    return path.exists() and path.is_file() and path.stat().st_size > 0


def check_environment() -> None:
    """
    Validate required environment variables.
    """
    print_header("STEP 1: ENVIRONMENT CHECK")

    env_path = PROJECT_ROOT / ".env"

    if not env_path.exists():
        raise FileNotFoundError(
            "\n.env file not found.\n\n"
            "Create a .env file in the project root with:\n\n"
            "OPENAI_API_KEY=your_openai_key_here\n"
            "DATABASE_URL=postgresql://postgres:postgres@localhost:5432/wardrobe_db\n"
        )

    openai_key = os.getenv("OPENAI_API_KEY")
    database_url = os.getenv("DATABASE_URL")

    if not openai_key:
        raise ValueError(
            "OPENAI_API_KEY is missing in .env file."
        )

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing in .env file."
        )

    print("Environment looks good.")
    print(f"Using DATABASE_URL: {database_url}")


def check_uploaded_images() -> None:
    """
    Make sure uploaded wardrobe images are present.
    """
    print_header("STEP 2: UPLOADED IMAGE CHECK")

    UPLOADED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    image_files = []
    for extension in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_files.extend(UPLOADED_IMAGES_DIR.glob(extension))

    if not image_files:
        raise FileNotFoundError(
            "\nNo wardrobe images found.\n\n"
            f"Please add images inside:\n{UPLOADED_IMAGES_DIR}\n\n"
            "Then run:\npython main.py"
        )

    print(f"Found {len(image_files)} wardrobe image(s).")
    print(f"Image folder: {UPLOADED_IMAGES_DIR}")


def check_postgres_connection() -> int:
    """
    Check PostgreSQL + pgvector DB connection.
    """
    print_header("STEP 3: POSTGRESQL / PGVECTOR CHECK")

    try:
        item_count = get_item_count()
        print("PostgreSQL connection is working.")
        print(f"Current wardrobe items in pgvector DB: {item_count}")
        return item_count

    except Exception as error:
        raise RuntimeError(
            "\nCould not connect to PostgreSQL / pgvector.\n\n"
            "Make sure Docker is running and start pgvector with:\n\n"
            "docker run --name wardrobe-pgvector `\n"
            "  -e POSTGRES_USER=postgres `\n"
            "  -e POSTGRES_PASSWORD=postgres `\n"
            "  -e POSTGRES_DB=wardrobe_db `\n"
            "  -p 5432:5432 `\n"
            "  -d pgvector/pgvector:pg16\n\n"
            "If container already exists, start it with:\n\n"
            "docker start wardrobe-pgvector\n"
        ) from error


def build_local_wardrobe_files_if_needed() -> None:
    """
    Build JSON files only if they are missing.

    This avoids re-captioning and re-embedding every time.
    """
    print_header("STEP 4: LOCAL WARDROBE PIPELINE CHECK")

    if file_exists(WARDROBE_INVENTORY_PATH):
        print(f"Found: {WARDROBE_INVENTORY_PATH}")
    else:
        print("wardrobe_inventory.json not found. Creating it...")
        run_script_module("scripts.check_uploaded_images")

    if file_exists(WARDROBE_METADATA_PATH):
        print(f"Found: {WARDROBE_METADATA_PATH}")
    else:
        print("wardrobe_metadata.json not found. Creating captions + metadata...")
        run_script_module("scripts.ingest_sample_closet")

    if file_exists(WARDROBE_TEXT_EMBEDDINGS_PATH):
        print(f"Found: {WARDROBE_TEXT_EMBEDDINGS_PATH}")
    else:
        print("wardrobe_embeddings.json not found. Creating text embeddings...")
        run_script_module("scripts.run_api")

    if file_exists(WARDROBE_HYBRID_EMBEDDINGS_PATH):
        print(f"Found: {WARDROBE_HYBRID_EMBEDDINGS_PATH}")
    else:
        print("wardrobe_hybrid_embeddings.json not found. Creating visual embeddings...")
        run_script_module("scripts.build_hybrid_embeddings")

    print("\nLocal wardrobe files are ready.")


def migrate_to_pgvector_if_needed(current_db_count: int) -> None:
    """
    Migrate wardrobe embeddings to pgvector only if DB is empty.
    """
    print_header("STEP 5: PGVECTOR MIGRATION CHECK")

    if current_db_count > 0:
        print(f"pgvector already has {current_db_count} item(s). Skipping migration.")
        return

    if not file_exists(WARDROBE_HYBRID_EMBEDDINGS_PATH):
        raise FileNotFoundError(
            "Cannot migrate to pgvector because wardrobe_hybrid_embeddings.json is missing."
        )

    print("pgvector DB is empty. Migrating wardrobe embeddings to PostgreSQL...")
    run_script_module("scripts.migrate_json_to_pgvector")

    new_count = get_item_count()
    print(f"Migration complete. Total items in pgvector DB: {new_count}")


def setup_chat_memory() -> None:
    """
    Create PostgreSQL chat memory tables.
    """
    print_header("STEP 6: CHAT MEMORY CHECK")

    create_chat_memory_schema()

    print("PostgreSQL chat memory tables are ready.")
    print("Using tables: chat_sessions, chat_messages, conversation_state")


def start_chat() -> None:
    """
    Start interactive fashion chatbot.
    """
    print_header("STEP 7: STARTING FASHION CHAT ASSISTANT")

    assistant = FashionChatAssistant()

    print("Fashion QnA Chat Assistant")
    print("=" * 90)
    print("Type your fashion request.")
    print()
    print("Examples:")
    print("- i have office tomorrow what should i wear")
    print("- style item_003 for casule day out")
    print("- What can I pair with item_003 for a smart casual outing?")
    print(
        "- What can I pair with "
        "C:\\Users\\Wardrobe_Recommendation_Engine\\11.jpg "
        "for a party?"
    )
    print("- Find similar items to what we have now")
    print()
    print("Type 'exit' to stop.")
    print("=" * 90)

    while True:
        user_message = input("\nYou: ").strip()

        if user_message.lower() in ["exit", "quit", "q"]:
            print("Assistant: Done. See you!")
            break

        if not user_message:
            continue

        try:
            assistant_response = assistant.chat(user_message)
            print("\nAssistant:", assistant_response)

        except Exception as error:
            print("\nAssistant: Something went wrong.")
            print(f"Error: {error}")


def main() -> None:
    """
    Main entrypoint for the complete Wardrobe Recommendation Engine.

    Recruiter/demo flow:
    1. Install requirements
    2. Setup .env
    3. Start Docker pgvector
    4. Run: python main.py
    """
    print_header("WARDROBE RECOMMENDATION ENGINE")
    print("Main pipeline started.")
    print(f"Project root: {PROJECT_ROOT}")

    check_environment()
    check_uploaded_images()

    db_count = check_postgres_connection()

    if db_count == 0:
        build_local_wardrobe_files_if_needed()

    migrate_to_pgvector_if_needed(current_db_count=db_count)
    setup_chat_memory()
    start_chat()


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
        sys.exit(0)

    except Exception as error:
        print("\n" + "=" * 90)
        print("MAIN PIPELINE FAILED")
        print("=" * 90)
        print(error)
        sys.exit(1)