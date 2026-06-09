from src.db.postgres_chat_memory import PostgresChatMemory


def main() -> None:
    memory = PostgresChatMemory(
        session_id="demo_pg_chat_session",
        user_id="demo_user",
        title="Wardrobe recommendation test chat",
    )

    memory.clear()

    print("=" * 80)
    print("POSTGRES CHAT MEMORY CHECK")
    print("=" * 80)

    memory.add_user_message(
        content="I have office tomorrow, what should I wear?",
        metadata={
            "intent": "text_outfit",
            "occasion": "office",
        },
    )

    memory.add_assistant_message(
        content="What kind of vibe are you looking for?",
        metadata={
            "needs_followup": True,
        },
    )

    memory.update_state(
        last_intent="text_outfit",
        last_occasion="office",
        last_vibe=None,
        last_user_query="I have office tomorrow, what should I wear?",
        last_assistant_response="What kind of vibe are you looking for?",
        context={
            "pending_followup": True,
            "occasion": "office",
        },
    )

    memory.add_user_message(
        content="smart clean look",
        metadata={
            "is_followup": True,
        },
    )

    memory.update_state(
        last_intent="text_outfit",
        last_occasion="office",
        last_vibe="smart clean",
        last_user_query="smart clean look",
        context={
            "pending_followup": False,
            "occasion": "office",
            "vibe": "smart clean",
        },
    )

    messages = memory.get_messages(limit=10)
    state = memory.get_state()
    context_text = memory.get_recent_context_text(limit=5)

    print("\nMessages:")
    for message in messages:
        print(f"- {message['role']}: {message['content']}")
        print(f"  metadata: {message['metadata']}")

    print("\nState:")
    print(state)

    print("\nRecent context text:")
    print(context_text)

    print("\nPostgres chat memory is working.")


if __name__ == "__main__":
    main()