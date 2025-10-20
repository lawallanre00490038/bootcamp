from tools import agent

def chatbot_agentic_rag():
    print("Agentic RAG Chatbot is running! Type 'exit' to quit.")
    while True:
        user_query = input("You: ")
        if user_query.lower() == "exit":
            print("Chatbot session ended.")
            break
        try:
            response = agent.run(user_query)
            print(f"Bot: {response}")
        except Exception as e:
            print(f"Error: {e}")