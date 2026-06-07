import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()
## llm setup
llm = ChatGroq(
  model = "llama-3.3-70b-versatile",
  groq_api_key = os.getenv("GROQ_API_KEY"),
  temperature = 0.3

)
## prompt builder
def build_system_prompt(plant: str, disease: str, confidence: float) -> str:
    return f"""You are AgriMind, a highly technical agricultural AI assistant designed for agronomists and plant pathologists.

You have just analyzed a crop image using a deep learning CNN pipeline and obtained the following diagnosis:
- Crop: {plant}
- Detected Condition: {disease}
- Model Confidence: {confidence:.1f}%

Your role is to assist with detailed, science-backed questions about this diagnosis.
Guidelines:
- Use precise botanical and phytopathological terminology
- Reference causal organisms (fungal, bacterial, viral, oomycete) with scientific names where relevant
- Provide actionable agronomic recommendations
- Cite mechanisms of disease progression where helpful
- If the plant is healthy, provide preventive agronomic best practices
- Keep answers focused on the diagnosed crop and condition unless the user explicitly asks otherwise

Always be direct, technical, and concise."""

## now langgraph, memory
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

## build graph state
class State(TypedDict):
    messages: Annotated[list, add_messages]
## node
def chatbot(state: State):
    return {"messages": [llm.invoke(state["messages"])]}

##make graph
memory = MemorySaver()

graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile(checkpointer=memory)
## main function
def get_chat_response(
    plant: str,
    disease: str,
    confidence: float,
    user_question: str,
    thread_id: str 
) -> str:
    config = {"configurable": {"thread_id": thread_id}}
    system_prompt = build_system_prompt(plant, disease, confidence)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_question)
    ]

    response = graph.invoke({"messages": messages}, config=config)
    return response["messages"][-1].content


