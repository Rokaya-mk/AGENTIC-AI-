from typing_extensions import TypedDict 
from typing import Literal 
from langgraph.graph import StateGraph, START, END 

class State(TypedDict): 
    topic: str 
    joke: str 
    improved: str 

def generate_joke(state: State): 
    return {"joke": f"A joke about {state['topic']} "} 

def improve_joke(state: State): 
    return {"improved": state['joke'] + " !!!"} 

def check_joke(state: State) -> Literal["improve", "end"]: 
    if "!" in state["joke"]: 
        return "end" 
    return "improve" 

builder = StateGraph(State) 
builder.add_node("generate_joke", generate_joke) 
builder.add_node("improve_joke", improve_joke) 

builder.add_edge(START, "generate_joke")
builder.add_conditional_edges("generate_joke", check_joke, {"improve": "improve_joke", "end": END}) 
builder.add_edge("improve_joke", END)

graph = builder.compile()

result = graph.invoke({"topic": "ice cream", "joke": "", "improved": ""})

print(result)