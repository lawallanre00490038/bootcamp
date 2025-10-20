from langchain.tools import Tool
from langchain.agents import initialize_agent, Tool, AgentType
from retrieve import retrieval_qa_chain
from model import llm

# Example: Weather API function
def get_query_response(query: str):
    return f"The response to your question {query} is: "

# Define the tool
query_tool = Tool(
    name="Query Tool",
    func=get_query_response,
    description="Provides the response to the question given"
)


# Combine tools and retrieval chain
tools = [
    Tool(
        name="Document Retrieval",
        func=lambda q: retrieval_qa_chain({"query": q})["result"],
        description="Retrieve knowledge from the document database."
    ),
    query_tool
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
