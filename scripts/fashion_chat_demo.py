from src.engine.fashion_chat_assistant import FashionChatAssistant


def main():
    print("Fashion QnA Chat Assistant")
    print("=" * 80)
    print("Type your fashion request.")
    print("Examples:")
    print("- I need to go to a party tonight")
    print("- I want a bold dark look")
    print("- What can I pair with item_003 for a smart casual outing?")
    print("- What can I pair with C:\\Users\\KIIT\\PycharmProjects\\Wardrobe_Recommendation_Engine\\11.jpg for a party?")
    print("- Find similar items to C:\\Users\\KIIT\\PycharmProjects\\Wardrobe_Recommendation_Engine\\11.jpg")
    print()
    print("Type 'exit' to stop.")
    print("=" * 80)

    assistant = FashionChatAssistant()

    while True:
        user_message = input("\nYou: ").strip()

        if user_message.lower() in ["exit", "quit", "q"]:
            print("Assistant: Done. See you!")
            break

        if not user_message:
            print("Assistant: Please type a fashion request.")
            continue

        try:
            assistant_response = assistant.chat(user_message)
            print(f"\nAssistant: {assistant_response}")

        except Exception as error:
            print(f"\nAssistant: Something went wrong: {error}")


if __name__ == "__main__":
    main()