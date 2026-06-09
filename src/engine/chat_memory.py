# import json
# from datetime import datetime
# from uuid import uuid4
#
# from src.utils import resolve_project_path
#
#
# CHAT_MEMORY_DIR = "data/chat_memory"
#
#
# class ChatMemory:
#     """
#     Simple local JSON chat memory.
#
#     Stores:
#     - user messages
#     - assistant messages
#     - extracted fashion context
#     - last recommendation
#     """
#
#     def __init__(self, session_id: str | None = None):
#         self.session_id = session_id or str(uuid4())
#         self.memory_dir = resolve_project_path(CHAT_MEMORY_DIR)
#         self.memory_dir.mkdir(parents=True, exist_ok=True)
#
#         self.memory_path = self.memory_dir / f"{self.session_id}.json"
#
#         self.data = {
#             "session_id": self.session_id,
#             "created_at": datetime.now().isoformat(),
#             "updated_at": datetime.now().isoformat(),
#             "context": {
#                 "pending_action": None,
#                 "occasion": None,
#                 "vibe": None,
#                 "color_preference": None,
#                 "item_id": None,
#                 "image_path": None,
#                 "last_user_request": None,
#             },
#             "messages": [],
#             "last_recommendation": None,
#         }
#
#         self.load()
#
#     def load(self) -> None:
#         """
#         Load previous session memory if it exists.
#         """
#         if self.memory_path.exists():
#             with self.memory_path.open("r", encoding="utf-8") as file:
#                 self.data = json.load(file)
#
#     def save(self) -> None:
#         """
#         Save session memory.
#         """
#         self.data["updated_at"] = datetime.now().isoformat()
#
#         with self.memory_path.open("w", encoding="utf-8") as file:
#             json.dump(self.data, file, indent=2, ensure_ascii=False)
#
#     def add_message(self, role: str, content: str) -> None:
#         """
#         Add user/assistant message.
#         """
#         self.data["messages"].append(
#             {
#                 "role": role,
#                 "content": content,
#                 "timestamp": datetime.now().isoformat(),
#             }
#         )
#         self.save()
#
#     def get_context(self) -> dict:
#         """
#         Get current extracted context.
#         """
#         return self.data.get("context", {})
#
#     def update_context(self, updates: dict) -> None:
#         """
#         Update fashion context.
#         """
#         context = self.data.get("context", {})
#
#         for key, value in updates.items():
#             if value is not None and value != "":
#                 context[key] = value
#
#         self.data["context"] = context
#         self.save()
#
#     def set_last_recommendation(self, recommendation: dict) -> None:
#         """
#         Store last recommendation for follow-up messages.
#         """
#         self.data["last_recommendation"] = recommendation
#         self.save()
#
#     def get_last_recommendation(self) -> dict | None:
#         """
#         Get previous recommendation.
#         """
#         return self.data.get("last_recommendation")
#
#     def reset_context(self) -> None:
#         """
#         Reset only active context, not full chat history.
#         """
#         self.data["context"] = {
#             "pending_action": None,
#             "occasion": None,
#             "vibe": None,
#             "color_preference": None,
#             "item_id": None,
#             "image_path": None,
#             "last_user_request": None,
#         }
#         self.save()


from uuid import uuid4

from src.db.postgres_chat_memory import PostgresChatMemory, make_json_safe


DEFAULT_CONTEXT = {
    "pending_action": None,
    "occasion": None,
    "vibe": None,
    "color_preference": None,
    "item_id": None,
    "image_path": None,
    "last_user_request": None,
    "raw_user_request": None,
    "last_active_source": None,
}


class ChatMemory:
    """
    Production-style chat memory adapter.

    This class keeps the same interface as the old local JSON ChatMemory,
    but stores everything in PostgreSQL using:

    - chat_sessions
    - chat_messages
    - conversation_state

    This lets existing chatbot code keep working without changes.
    """

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or f"fashion_session_{uuid4().hex[:12]}"

        self.pg_memory = PostgresChatMemory(
            session_id=self.session_id,
            user_id="demo_user",
            title="Wardrobe recommendation chat",
        )

        self._ensure_default_context()

    def _ensure_default_context(self) -> None:
        """
        Make sure the session has a context object.
        """
        state = self.pg_memory.get_state()
        context = state.get("context", {}) if state else {}

        if not context:
            self.pg_memory.update_state(
                context=DEFAULT_CONTEXT,
            )

    def _get_state(self) -> dict:
        """
        Internal helper to fetch PostgreSQL state.
        """
        return self.pg_memory.get_state() or {}

    def add_message(self, role: str, content: str) -> None:
        """
        Add user/assistant message.

        This matches the old JSON ChatMemory method signature.
        """
        self.pg_memory.add_message(
            role=role,
            content=content,
            metadata={},
        )

    def get_context(self) -> dict:
        """
        Get current extracted context.
        """
        state = self._get_state()
        context = state.get("context") or {}

        merged_context = DEFAULT_CONTEXT.copy()
        merged_context.update(context)

        return merged_context

    def update_context(self, updates: dict) -> None:
        """
        Update fashion context.

        Same behavior as the old JSON version:
        only non-empty values are updated.
        """
        current_context = self.get_context()

        for key, value in updates.items():
            if value is not None and value != "":
                current_context[key] = value

        current_context = make_json_safe(current_context)

        self.pg_memory.update_state(
            last_intent=current_context.get("pending_action"),
            last_item_id=current_context.get("item_id"),
            last_image_path=current_context.get("image_path"),
            last_occasion=current_context.get("occasion"),
            last_vibe=current_context.get("vibe"),
            last_user_query=current_context.get("last_user_request"),
            context=current_context,
        )

    def set_last_recommendation(self, recommendation: dict) -> None:
        """
        Store last recommendation for follow-up messages.
        """
        safe_recommendation = make_json_safe(recommendation)

        state = self._get_state()
        context = state.get("context") or self.get_context()

        self.pg_memory.update_state(
            last_intent=context.get("pending_action"),
            last_item_id=context.get("item_id"),
            last_image_path=context.get("image_path"),
            last_occasion=context.get("occasion"),
            last_vibe=context.get("vibe"),
            last_user_query=context.get("last_user_request"),
            last_recommendation=safe_recommendation,
            context=context,
        )

    def get_last_recommendation(self) -> dict | None:
        """
        Get previous recommendation.
        """
        state = self._get_state()
        recommendation = state.get("last_recommendation")

        if recommendation:
            return recommendation

        return None

    def reset_context(self) -> None:
        """
        Reset only active context, not full chat history.
        """
        self.pg_memory.update_state(
            last_intent=None,
            last_item_id=None,
            last_image_path=None,
            last_occasion=None,
            last_vibe=None,
            last_user_query=None,
            context=DEFAULT_CONTEXT,
        )

    def get_messages(self, limit: int = 20) -> list[dict]:
        """
        Optional helper to fetch recent chat messages.
        """
        return self.pg_memory.get_messages(limit=limit)

    def get_recent_context_text(self, limit: int = 8) -> str:
        """
        Optional helper to fetch recent chat messages as text.
        """
        return self.pg_memory.get_recent_context_text(limit=limit)

    def clear(self) -> None:
        """
        Clear current session messages and state.
        """
        self.pg_memory.clear()
        self._ensure_default_context()