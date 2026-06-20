from typing_extensions import TypedDict 
from typing import Literal 
from langgraph.graph import StateGraph, START, END 
from IPython.display import Image, display 

class State(TypedDict): 
    n: int 
    log: list[str] 

def step(state: State): 
    n = state["n"] + 1 
    return {"n": n, "log": state["log"] + [f"n is now {n}"]} 
def should_continue(state: State) -> Literal["again", "stop"]: 
    return "again" if state["n"] < 5 else "stop" 

builder = StateGraph(State) 
builder.add_node("step", step) 
builder.add_edge(START, "step") 
builder.add_conditional_edges("step", should_continue, {"again": "step", "stop": END}) # START -> step n + 0 = 1, -> step n + 1 = 2 -> END 
graph = builder.compile() 
result = graph.invoke({"n": 0, "log": []}) 
print(result) 
png_bytes = graph.get_graph().draw_mermaid_png() 
with open("graph2.png", "wb") as f: 
    f.write(png_bytes)