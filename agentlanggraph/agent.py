from operator import add 
from typing import Literal 
from typing_extensions import TypedDict, Annotated 
from langchain_core.messages import AnyMessage 
from langchain.messages import SystemMessage, ToolMessage, HumanMessage 
from langgraph.graph import StateGraph, START, END 
from tools_setup import model_with_tools, tools_by_name


class AgentState(TypedDict): 
    messages: Annotated[list[AnyMessage], add] 
    llm_calls: int

def llm_call(state: AgentState): 
    """LLM decides whether to call tools or respond.""" 
    response = model_with_tools.invoke(
        [SystemMessage(content="You are a helpful assistant that solves arith-metic problem using tools when needed.")] 
        + state["messages"] 
        )
    
    return { 
        "messages": [response], 
        "llm_calls": state.get("llm_calls", 0) + 1, 
        }

def tool_node(state: AgentState): 
    """Execute tool calls from last AI message.""" 
    last = state["messages"][-1] 
    results = [] 
    for call in last.tool_calls: 
        tool = tools_by_name[call["name"]] 
        observation = tool.invoke(call["args"]) 
        results.append(ToolMessage(content=str(observation), 
                                   tool_call_id=call["id"])) 
        return {"messages": results}

def should_continue(state: AgentState) -> Literal["tool_node", END]: 
    last = state["messages"][-1] 
    if getattr(last, "tool_calls", None): 
        if last.tool_calls: 
            return "tool_node" 
    return END

builder = StateGraph(AgentState) 

builder.add_node("llm_call", llm_call) 
builder.add_node("tool_node", tool_node) 

builder.add_edge(START, "llm_call") 
builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END]) 
builder.add_edge("tool_node", "llm_call")

agent = builder.compile()

if __name__ == "__main__": 
    result = agent.invoke({"messages": [HumanMessage(content="Add 3 and 4.")], "llm_calls": 0})

    print(result) 
    for m in result["messages"]: # parcourir tout l’historique conversationnel 
        try: 
            m.pretty_print() # affiche type du message, contenu et tool calls 
        except Exception as ex: 
            print(ex) 
            print(m)

# Stream updates (state deltas) 
    for chunk in agent.stream( # chaque chunk est une modification du state 
        {"messages": [HumanMessage(content="Multiple 30 and 43.")], "llm_calls": 0}, 
        stream_mode="updates",#afficher chaque modification du graphe en direct node par node 
        ): 
        print(chunk)
    for message_chunk, metadata in agent.stream(
        {"messages": [HumanMessage(content="Divide 30 and 43.")], "llm_calls": 0}, 
        stream_mode="messages", #les sorties du LLM et métadonnées 
        ): 
        if getattr(message_chunk, "content", None): 
            print(message_chunk.content, end="", flush=True) 
    print()