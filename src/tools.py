from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool, AgentType
from retrieve import retrieval_qa_chain
from model import llm

# === Define Safe Tool Functions ===
def get_query_response(query: str) -> str:
    try:
        if not query.strip():
            return "No query provided."
        return f"The response to your question '{query}' is: [placeholder answer]"
    except Exception as e:
        return f"Error in Query Tool: {e}"

def get_retrieval_answer(query: str) -> str:
    try:
        if not query.strip():
            return "No query provided for document retrieval."
        result = retrieval_qa_chain({"query": query})
        if isinstance(result, dict):
            return str(result.get("result", "No result found."))
        return str(result)
    except Exception as e:
        return f"Document retrieval error: {e}"

# === Define Tools ===
tools = [
    Tool(
        name="Document Retrieval",
        func=get_retrieval_answer,
        description="Use this to retrieve answers from uploaded PDFs."
    ),
    Tool(
        name="Query Tool",
        func=get_query_response,
        description="Use this to handle general knowledge questions."
    ),
]


# Initialize the agent
# === Initialize the Agent ===
print("Initializing agent...")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=3,  # Prevent endless loops
)