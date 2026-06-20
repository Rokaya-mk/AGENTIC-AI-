from typing_extensions import TypedDict 
from langgraph.graph import StateGraph, START, END

class State(TypedDict): 
    topic: str 
    joke: str

def refine_topic(state: State): 
    return {"topic": state["topic"] + " (and cats)"}

def write_joke(state: State): 
    return {"joke": f"Here is a joke about {state['topic']}."}
builder = StateGraph(State) 
builder.add_node("refine_topic", refine_topic) 
builder.add_node("write_joke", write_joke) 
builder.add_edge(START, "refine_topic") 
builder.add_edge("refine_topic", "write_joke") 
builder.add_edge("write_joke", END) 
graph = builder.compile() 
result = graph.invoke({"topic": "ice cream", "joke": ""}) 
print(result)