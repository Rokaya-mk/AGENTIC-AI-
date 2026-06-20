from typing_extensions import TypedDict, Annotated 
from operator import add 
from langchain_core.messages import BaseMessage 
from langgraph.graph import StateGraph, START, END

class ChatState(TypedDict): 
    messages: Annotated[list[BaseMessage], add] 
    steps: int

def echo_node(state: ChatState): 
    return { "messages": [
        {"role": "ai",
          "content": f"Step {state['steps']}: got your message."
          }
          ], 
          "steps": state["steps"] + 1,
         }

def echo_node_1(state: ChatState): 
    return { 
        "messages": [
            {
                "role": "ai", 
                "content": f"Step {state['steps']}: got your message 1."
                }
            ], 
        "steps": state["steps"] + 1, 
        }
builder = StateGraph(ChatState) 
builder.add_node("echo", echo_node) 
builder.add_node("echo_1", echo_node_1) 
builder.add_edge(START, "echo") 
builder.add_edge("echo", "echo_1") 
builder.add_edge("echo_1", END) 
graph = builder.compile() 
result = graph.invoke({"messages": [{"role": "user", "content": f"hello"}], "steps": 1}) 
print(result["messages"])