import json
from datetime import datetime
from uuid import uuid4

from src.db.pgvector_store import get_connection


def make_json_safe(value):
    """
    Convert values into JSON-safe format.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}

    if isinstance(value, list):
        return [make_json_safe(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]

    if isinstance(value, set):
        return [make_json_safe(item) for item in value]

    if hasattr(value, "item"):
        return make_json_safe(value.item())

    if isinstance(value, (str, int, float, bool)):
        return value

    return str(value)


def create_chat_memory_schema() -> None:
    """
    Create production-style chat memory tables in PostgreSQL.

    Tables:
    - chat_sessions: one row per chat session
    - chat_messages: all user/assistant messages
    - conversation_state: latest useful memory for follow-up handling
    """
    with get_connection(register_pgvector=True) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS chat_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES chat_sessions(session_id)
                    ON DELETE CASCADE,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_state (
                    session_id TEXT PRIMARY KEY REFERENCES chat_sessions(session_id)
                    ON DELETE CASCADE,
                    last_intent TEXT,
                    last_item_id TEXT,
                    last_image_path TEXT,
                    last_occasion TEXT,
                    last_vibe TEXT,
                    last_user_query TEXT,
                    last_assistant_response TEXT,
                    last_recommendation JSONB DEFAULT '{}'::jsonb,
                    context JSONB DEFAULT '{}'::jsonb,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chat_messages_session_idx
                ON chat_messages (session_id, created_at);
                """
            )

            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS chat_sessions_user_idx
                ON chat_sessions (user_id);
                """
            )

        conn.commit()


class PostgresChatMemory:
    """
    PostgreSQL-backed chat memory.

    This replaces local JSON memory later.
    For now, we first test it independently.
    """

    def __init__(
        self,
        session_id: str | None = None,
        user_id: str = "demo_user",
        title: str | None = None,
    ) -> None:
        create_chat_memory_schema()

        self.session_id = session_id or f"session_{uuid4().hex[:12]}"
        self.user_id = user_id
        self.title = title or "Fashion chat session"

        self.create_or_touch_session()

    def create_or_touch_session(self) -> None:
        """
        Create session if missing, otherwise update timestamp.
        """
        with get_connection(register_pgvector=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_sessions (
                        session_id,
                        user_id,
                        title,
                        created_at,
                        updated_at
                    )
                    VALUES (
                        %(session_id)s,
                        %(user_id)s,
                        %(title)s,
                        %(now)s,
                        %(now)s
                    )
                    ON CONFLICT (session_id)
                    DO UPDATE SET
                        updated_at = EXCLUDED.updated_at;
                    """,
                    {
                        "session_id": self.session_id,
                        "user_id": self.user_id,
                        "title": self.title,
                        "now": datetime.now(),
                    },
                )

            conn.commit()

    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict | None = None,
    ) -> str:
        """
        Store one chat message.

        role should usually be:
        - user
        - assistant
        - system
        """
        message_id = f"msg_{uuid4().hex}"

        safe_metadata = make_json_safe(metadata or {})

        with get_connection(register_pgvector=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO chat_messages (
                        message_id,
                        session_id,
                        role,
                        content,
                        metadata,
                        created_at
                    )
                    VALUES (
                        %(message_id)s,
                        %(session_id)s,
                        %(role)s,
                        %(content)s,
                        %(metadata)s::jsonb,
                        %(created_at)s
                    );
                    """,
                    {
                        "message_id": message_id,
                        "session_id": self.session_id,
                        "role": role,
                        "content": content,
                        "metadata": json.dumps(safe_metadata),
                        "created_at": datetime.now(),
                    },
                )

                cur.execute(
                    """
                    UPDATE chat_sessions
                    SET updated_at = CURRENT_TIMESTAMP
                    WHERE session_id = %(session_id)s;
                    """,
                    {"session_id": self.session_id},
                )

            conn.commit()

        return message_id

    def add_user_message(
        self,
        content: str,
        metadata: dict | None = None,
    ) -> str:
        """
        Store user message.
        """
        return self.add_message(
            role="user",
            content=content,
            metadata=metadata,
        )

    def add_assistant_message(
        self,
        content: str,
        metadata: dict | None = None,
    ) -> str:
        """
        Store assistant message.
        """
        return self.add_message(
            role="assistant",
            content=content,
            metadata=metadata,
        )

    def get_messages(self, limit: int = 20) -> list[dict]:
        """
        Fetch latest messages for this session.
        """
        with get_connection(register_pgvector=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        message_id,
                        role,
                        content,
                        metadata,
                        created_at
                    FROM chat_messages
                    WHERE session_id = %(session_id)s
                    ORDER BY created_at DESC
                    LIMIT %(limit)s;
                    """,
                    {
                        "session_id": self.session_id,
                        "limit": limit,
                    },
                )

                rows = cur.fetchall()

        messages = []

        for row in reversed(rows):
            messages.append(
                {
                    "message_id": row[0],
                    "role": row[1],
                    "content": row[2],
                    "metadata": row[3] or {},
                    "created_at": str(row[4]),
                }
            )

        return messages

    def get_recent_context_text(self, limit: int = 8) -> str:
        """
        Return recent messages as simple context text for LLM prompts.
        """
        messages = self.get_messages(limit=limit)

        lines = []

        for message in messages:
            role = message.get("role", "unknown")
            content = message.get("content", "")
            lines.append(f"{role}: {content}")

        return "\n".join(lines)

    def update_state(
        self,
        last_intent: str | None = None,
        last_item_id: str | None = None,
        last_image_path: str | None = None,
        last_occasion: str | None = None,
        last_vibe: str | None = None,
        last_user_query: str | None = None,
        last_assistant_response: str | None = None,
        last_recommendation: dict | None = None,
        context: dict | None = None,
    ) -> None:
        """
        Store latest structured conversation state.

        This is what helps follow-up queries like:
        - make it more casual
        - find similar items to what we have now
        - use the same image
        """
        safe_recommendation = make_json_safe(last_recommendation or {})
        safe_context = make_json_safe(context or {})

        with get_connection(register_pgvector=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversation_state (
                        session_id,
                        last_intent,
                        last_item_id,
                        last_image_path,
                        last_occasion,
                        last_vibe,
                        last_user_query,
                        last_assistant_response,
                        last_recommendation,
                        context,
                        updated_at
                    )
                    VALUES (
                        %(session_id)s,
                        %(last_intent)s,
                        %(last_item_id)s,
                        %(last_image_path)s,
                        %(last_occasion)s,
                        %(last_vibe)s,
                        %(last_user_query)s,
                        %(last_assistant_response)s,
                        %(last_recommendation)s::jsonb,
                        %(context)s::jsonb,
                        CURRENT_TIMESTAMP
                    )
                    ON CONFLICT (session_id)
                    DO UPDATE SET
                        last_intent = COALESCE(EXCLUDED.last_intent, conversation_state.last_intent),
                        last_item_id = COALESCE(EXCLUDED.last_item_id, conversation_state.last_item_id),
                        last_image_path = COALESCE(EXCLUDED.last_image_path, conversation_state.last_image_path),
                        last_occasion = COALESCE(EXCLUDED.last_occasion, conversation_state.last_occasion),
                        last_vibe = COALESCE(EXCLUDED.last_vibe, conversation_state.last_vibe),
                        last_user_query = COALESCE(EXCLUDED.last_user_query, conversation_state.last_user_query),
                        last_assistant_response = COALESCE(EXCLUDED.last_assistant_response, conversation_state.last_assistant_response),
                        last_recommendation = COALESCE(EXCLUDED.last_recommendation, conversation_state.last_recommendation),
                        context = COALESCE(EXCLUDED.context, conversation_state.context),
                        updated_at = CURRENT_TIMESTAMP;
                    """,
                    {
                        "session_id": self.session_id,
                        "last_intent": last_intent,
                        "last_item_id": last_item_id,
                        "last_image_path": last_image_path,
                        "last_occasion": last_occasion,
                        "last_vibe": last_vibe,
                        "last_user_query": last_user_query,
                        "last_assistant_response": last_assistant_response,
                        "last_recommendation": json.dumps(safe_recommendation),
                        "context": json.dumps(safe_context),
                    },
                )

            conn.commit()

    def get_state(self) -> dict:
        """
        Fetch latest structured state.
        """
        with get_connection(register_pgvector=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT
                        last_intent,
                        last_item_id,
                        last_image_path,
                        last_occasion,
                        last_vibe,
                        last_user_query,
                        last_assistant_response,
                        last_recommendation,
                        context,
                        updated_at
                    FROM conversation_state
                    WHERE session_id = %(session_id)s;
                    """,
                    {"session_id": self.session_id},
                )

                row = cur.fetchone()

        if not row:
            return {}

        return {
            "last_intent": row[0],
            "last_item_id": row[1],
            "last_image_path": row[2],
            "last_occasion": row[3],
            "last_vibe": row[4],
            "last_user_query": row[5],
            "last_assistant_response": row[6],
            "last_recommendation": row[7] or {},
            "context": row[8] or {},
            "updated_at": str(row[9]),
        }

    def get_context(self) -> dict:
        """
        Compatibility helper.

        Existing code can call memory.get_context().
        """
        state = self.get_state()
        return state.get("context", {})

    def clear(self) -> None:
        """
        Delete this session's messages and state.
        """
        with get_connection(register_pgvector=True) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM chat_messages
                    WHERE session_id = %(session_id)s;
                    """,
                    {"session_id": self.session_id},
                )

                cur.execute(
                    """
                    DELETE FROM conversation_state
                    WHERE session_id = %(session_id)s;
                    """,
                    {"session_id": self.session_id},
                )

            conn.commit()