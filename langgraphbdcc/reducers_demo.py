from typing_extensions import TypedDict, Annotated 
from operator import add 
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
     topic: str 
     log: Annotated[list[str], add]
def step_a(state: State): 
    return {"log": [f"step_a saw topic={state['topic']}"]}

def step_b(state: State): 
    return {"log": [f"step_b finishing topic={state['topic']}"]} 

def step_c(state: State): 
    return {"log": [f"step_c finishing topic at last={state['topic']}"]}

builder = StateGraph(State) 
builder.add_node("step_a", step_a) 
builder.add_node("step_b", step_b) 
builder.add_node("step_c", step_c) 
builder.add_edge(START, "step_a") 
builder.add_edge("step_a", "step_b") 
builder.add_edge("step_b", "step_c") 
builder.add_edge("step_c", END) 
graph = builder.compile() 
result = graph.invoke({"topic": "ice cream", "log": []}) 
print(result['log'])