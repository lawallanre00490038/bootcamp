from tools import agent
# === Chatbot Loop ===
def chatbot_agentic_rag():
    print("\n🤖 Agentic RAG Chatbot is running! Type 'exit' to quit.\n")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Chatbot session ended.")
            break
        try:
            response = agent.run(user_query)
            print(f"Bot: {response}\n")
        except Exception as e:
            print(f"⚠️ Error: {e}\n")

if __name__ == "__main__":
    chatbot_agentic_rag()