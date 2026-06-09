import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from src.utils import resolve_project_path


CHAT_MEMORY_DIR = "data/chat_memory"


class ChatMemory:
    """
    Simple local JSON chat memory.

    Stores:
    - user messages
    - assistant messages
    - extracted fashion context
    - last recommendation
    """

    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid4())
        self.memory_dir = resolve_project_path(CHAT_MEMORY_DIR)
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        self.memory_path = self.memory_dir / f"{self.session_id}.json"

        self.data = {
            "session_id": self.session_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "context": {
                "pending_action": None,
                "occasion": None,
                "vibe": None,
                "color_preference": None,
                "item_id": None,
                "image_path": None,
                "last_user_request": None,
            },
            "messages": [],
            "last_recommendation": None,
        }

        self.load()

    def load(self) -> None:
        """
        Load previous session memory if it exists.
        """
        if self.memory_path.exists():
            with self.memory_path.open("r", encoding="utf-8") as file:
                self.data = json.load(file)

    def save(self) -> None:
        """
        Save session memory.
        """
        self.data["updated_at"] = datetime.now().isoformat()

        with self.memory_path.open("w", encoding="utf-8") as file:
            json.dump(self.data, file, indent=2, ensure_ascii=False)

    def add_message(self, role: str, content: str) -> None:
        """
        Add user/assistant message.
        """
        self.data["messages"].append(
            {
                "role": role,
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )
        self.save()

    def get_context(self) -> dict:
        """
        Get current extracted context.
        """
        return self.data.get("context", {})

    def update_context(self, updates: dict) -> None:
        """
        Update fashion context.
        """
        context = self.data.get("context", {})

        for key, value in updates.items():
            if value is not None and value != "":
                context[key] = value

        self.data["context"] = context
        self.save()

    def set_last_recommendation(self, recommendation: dict) -> None:
        """
        Store last recommendation for follow-up messages.
        """
        self.data["last_recommendation"] = recommendation
        self.save()

    def get_last_recommendation(self) -> dict | None:
        """
        Get previous recommendation.
        """
        return self.data.get("last_recommendation")

    def reset_context(self) -> None:
        """
        Reset only active context, not full chat history.
        """
        self.data["context"] = {
            "pending_action": None,
            "occasion": None,
            "vibe": None,
            "color_preference": None,
            "item_id": None,
            "image_path": None,
            "last_user_request": None,
        }
        self.save()