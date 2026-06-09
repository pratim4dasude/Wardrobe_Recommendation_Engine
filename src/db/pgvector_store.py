import json
import os
from typing import Any

import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector


load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/wardrobe_db",
)


def get_connection(register_pgvector: bool = True):
    """
    Create PostgreSQL connection.

    Important:
    During first schema creation, pgvector extension may not exist yet.
    So we allow connecting without register_vector().
    """
    conn = psycopg.connect(DATABASE_URL)

    if register_pgvector:
        register_vector(conn)

    return conn


def create_pgvector_schema() -> None:
    """
    Create pgvector extension and wardrobe_items table.
    """
    # Step 1: create extension without registering pgvector first.
    with get_connection(register_pgvector=False) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()

    # Step 2: now vector type exists, so register pgvector normally.
    with get_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS wardrobe_items (
                    id BIGSERIAL PRIMARY KEY,
                    item_id TEXT UNIQUE NOT NULL,
                    filename TEXT,
                    image_path TEXT,
                    caption TEXT,
                    search_text TEXT,
                    category JSONB,
                    color JSONB,
                    style JSONB,
                    item_type TEXT,
                    text_embedding vector(1536),
                    visual_embedding vector(512),
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS wardrobe_item_type_idx
                ON wardrobe_items (item_type);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS wardrobe_text_embedding_idx
                ON wardrobe_items
                USING ivfflat (text_embedding vector_cosine_ops)
                WITH (lists = 100);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS wardrobe_visual_embedding_idx
                ON wardrobe_items
                USING ivfflat (visual_embedding vector_cosine_ops)
                WITH (lists = 100);
                """
            )

        conn.commit()


def to_pgvector(vector: list[float]) -> Vector:
    """
    Convert Python list / numpy float values into pgvector Vector type.

    This fixes:
    operator does not exist: vector <=> double precision[]
    """
    clean_vector = [float(value) for value in vector]
    return Vector(clean_vector)


def detect_item_type_from_metadata(item: dict[str, Any]) -> str:
    """
    Simple item type detection for DB storage.
    """
    category = item.get("category", [])
    caption = item.get("caption", "")
    search_text = item.get("search_text", "")

    text = f"{category} {caption} {search_text}".lower()

    if any(word in text for word in ["shirt", "t-shirt", "tee", "top", "blouse"]):
        return "upperwear"

    if any(word in text for word in ["pants", "trousers", "jeans", "shorts"]):
        return "bottomwear"

    if any(
        word in text
        for word in ["jacket", "blazer", "hoodie", "sweater", "cardigan", "coat"]
    ):
        return "outerwear"

    if "dress" in text:
        return "dress"

    if any(word in text for word in ["shoe", "sneaker", "loafer", "boot"]):
        return "footwear"

    if any(word in text for word in ["watch", "belt", "cap", "hat", "bag"]):
        return "accessory"

    return "unknown"


def validate_vector_size(
    vector: list[float] | None,
    expected_size: int,
    vector_name: str,
    item_id: str,
) -> bool:
    """
    Validate embedding size before inserting into pgvector.
    """
    if not vector:
        print(f"Skipping {item_id}: missing {vector_name}")
        return False

    if len(vector) != expected_size:
        print(
            f"Skipping {item_id}: {vector_name} size is {len(vector)}, "
            f"expected {expected_size}"
        )
        return False

    return True


def upsert_wardrobe_item(item: dict[str, Any]) -> bool:
    """
    Insert or update one wardrobe item into PostgreSQL.
    """
    item_id = item.get("item_id")

    if not item_id:
        print("Skipping item with missing item_id")
        return False

    text_embedding = item.get("text_embedding")
    visual_embedding = item.get("visual_embedding")

    text_ok = validate_vector_size(
        vector=text_embedding,
        expected_size=1536,
        vector_name="text_embedding",
        item_id=item_id,
    )

    visual_ok = validate_vector_size(
        vector=visual_embedding,
        expected_size=512,
        vector_name="visual_embedding",
        item_id=item_id,
    )

    if not text_ok or not visual_ok:
        return False

    item_type = item.get("item_type") or detect_item_type_from_metadata(item)

    with get_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO wardrobe_items (
                    item_id,
                    filename,
                    image_path,
                    caption,
                    search_text,
                    category,
                    color,
                    style,
                    item_type,
                    text_embedding,
                    visual_embedding
                )
                VALUES (
                    %(item_id)s,
                    %(filename)s,
                    %(image_path)s,
                    %(caption)s,
                    %(search_text)s,
                    %(category)s::jsonb,
                    %(color)s::jsonb,
                    %(style)s::jsonb,
                    %(item_type)s,
                    %(text_embedding)s,
                    %(visual_embedding)s
                )
                ON CONFLICT (item_id)
                DO UPDATE SET
                    filename = EXCLUDED.filename,
                    image_path = EXCLUDED.image_path,
                    caption = EXCLUDED.caption,
                    search_text = EXCLUDED.search_text,
                    category = EXCLUDED.category,
                    color = EXCLUDED.color,
                    style = EXCLUDED.style,
                    item_type = EXCLUDED.item_type,
                    text_embedding = EXCLUDED.text_embedding,
                    visual_embedding = EXCLUDED.visual_embedding;
                """,
                {
                    "item_id": item_id,
                    "filename": item.get("filename"),
                    "image_path": item.get("image_path"),
                    "caption": item.get("caption"),
                    "search_text": item.get("search_text"),
                    "category": json.dumps(item.get("category", [])),
                    "color": json.dumps(item.get("color", [])),
                    "style": json.dumps(item.get("style", [])),
                    "item_type": item_type,
                    "text_embedding": to_pgvector(text_embedding),
                    "visual_embedding": to_pgvector(visual_embedding),
                },
            )

        conn.commit()

    return True


def clean_vector_values(vector) -> list[float] | None:
    """
    Convert pgvector / numpy vector values into normal Python floats.

    This prevents:
    Object of type float32 is not JSON serializable
    """
    if vector is None:
        return None

    return [float(value) for value in vector]


def row_to_item(row) -> dict[str, Any]:
    """
    Convert DB row into the same dict format used by the project.
    """
    return {
        "item_id": row[0],
        "filename": row[1],
        "image_path": row[2],
        "caption": row[3],
        "search_text": row[4],
        "category": row[5] or [],
        "color": row[6] or [],
        "style": row[7] or [],
        "item_type": row[8],
        "text_embedding": clean_vector_values(row[9]),
        "visual_embedding": clean_vector_values(row[10]),
    }


def fetch_item_by_id(item_id: str) -> dict[str, Any] | None:
    """
    Fetch one wardrobe item from pgvector DB.
    """
    with get_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    item_id,
                    filename,
                    image_path,
                    caption,
                    search_text,
                    category,
                    color,
                    style,
                    item_type,
                    text_embedding,
                    visual_embedding
                FROM wardrobe_items
                WHERE item_id = %s;
                """,
                (item_id,),
            )

            row = cur.fetchone()

    if not row:
        return None

    return row_to_item(row)


def load_all_items() -> list[dict[str, Any]]:
    """
    Load all wardrobe items from pgvector DB.
    This will later replace loading from wardrobe_hybrid_embeddings.json.
    """
    with get_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    item_id,
                    filename,
                    image_path,
                    caption,
                    search_text,
                    category,
                    color,
                    style,
                    item_type,
                    text_embedding,
                    visual_embedding
                FROM wardrobe_items
                ORDER BY item_id;
                """
            )

            rows = cur.fetchall()

    return [row_to_item(row) for row in rows]


def get_item_count() -> int:
    """
    Count items in pgvector DB.
    """
    with get_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM wardrobe_items;")
            count = cur.fetchone()[0]

    return int(count)


def search_by_text_embedding(
    query_embedding: list[float],
    top_k: int = 8,
    item_type: str | None = None,
    exclude_item_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search wardrobe items using text embedding stored in pgvector.
    """
    params = {
        "query_embedding": to_pgvector(query_embedding),
        "top_k": top_k,
    }

    filters = []

    if item_type:
        filters.append("item_type = %(item_type)s")
        params["item_type"] = item_type

    if exclude_item_id:
        filters.append("item_id != %(exclude_item_id)s")
        params["exclude_item_id"] = exclude_item_id

    where_clause = ""

    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    query = f"""
        SELECT
            item_id,
            filename,
            image_path,
            caption,
            search_text,
            category,
            color,
            style,
            item_type,
            text_embedding,
            visual_embedding,
            text_embedding <=> %(query_embedding)s AS distance
        FROM wardrobe_items
        {where_clause}
        ORDER BY text_embedding <=> %(query_embedding)s
        LIMIT %(top_k)s;
    """

    with get_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    results = []

    for row in rows:
        item = row_to_item(row[:11])
        distance = float(row[11])

        item["text_score"] = round(1 - distance, 4)
        item["visual_score"] = None
        item["hybrid_pgvector_score"] = None
        item["similarity_score"] = item["text_score"]

        results.append(item)

    return results


def search_by_visual_embedding(
    query_embedding: list[float],
    top_k: int = 8,
    item_type: str | None = None,
    exclude_item_id: str | None = None,
) -> list[dict[str, Any]]:
    """
    Search wardrobe items using visual embedding stored in pgvector.
    """
    params = {
        "query_embedding": to_pgvector(query_embedding),
        "top_k": top_k,
    }

    filters = []

    if item_type:
        filters.append("item_type = %(item_type)s")
        params["item_type"] = item_type

    if exclude_item_id:
        filters.append("item_id != %(exclude_item_id)s")
        params["exclude_item_id"] = exclude_item_id

    where_clause = ""

    if filters:
        where_clause = "WHERE " + " AND ".join(filters)

    query = f"""
        SELECT
            item_id,
            filename,
            image_path,
            caption,
            search_text,
            category,
            color,
            style,
            item_type,
            text_embedding,
            visual_embedding,
            visual_embedding <=> %(query_embedding)s AS distance
        FROM wardrobe_items
        {where_clause}
        ORDER BY visual_embedding <=> %(query_embedding)s
        LIMIT %(top_k)s;
    """

    with get_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()

    results = []

    for row in rows:
        item = row_to_item(row[:11])
        distance = float(row[11])

        item["text_score"] = None
        item["visual_score"] = round(1 - distance, 4)
        item["hybrid_pgvector_score"] = None
        item["similarity_score"] = item["visual_score"]

        results.append(item)

    return results


def hybrid_search_text_and_visual(
    text_embedding: list[float],
    visual_embedding: list[float],
    top_k: int = 8,
    item_type: str | None = None,
    exclude_item_id: str | None = None,
    text_weight: float = 0.40,
    visual_weight: float = 0.60,
) -> list[dict[str, Any]]:
    """
    Search text and visual separately, then combine scores.

    This is the main hybrid retrieval formula:

    final_score = text_weight * text_score + visual_weight * visual_score
    """
    text_results = search_by_text_embedding(
        query_embedding=text_embedding,
        top_k=top_k * 3,
        item_type=item_type,
        exclude_item_id=exclude_item_id,
    )

    visual_results = search_by_visual_embedding(
        query_embedding=visual_embedding,
        top_k=top_k * 3,
        item_type=item_type,
        exclude_item_id=exclude_item_id,
    )

    merged: dict[str, dict[str, Any]] = {}

    for item in text_results:
        item_id = item["item_id"]
        merged[item_id] = item
        merged[item_id]["text_score"] = item.get("text_score") or 0.0
        merged[item_id]["visual_score"] = 0.0

    for item in visual_results:
        item_id = item["item_id"]

        if item_id not in merged:
            merged[item_id] = item
            merged[item_id]["text_score"] = 0.0

        merged[item_id]["visual_score"] = item.get("visual_score") or 0.0

    results = []

    for item in merged.values():
        text_score = item.get("text_score") or 0.0
        visual_score = item.get("visual_score") or 0.0

        final_score = (text_weight * text_score) + (visual_weight * visual_score)

        item["text_score"] = round(text_score, 4)
        item["visual_score"] = round(visual_score, 4)
        item["hybrid_pgvector_score"] = round(final_score, 4)
        item["similarity_score"] = round(final_score, 4)

        results.append(item)

    results.sort(key=lambda item: item["similarity_score"], reverse=True)

    return results[:top_k]