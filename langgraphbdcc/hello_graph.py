from langgraph.graph import StateGraph,MessagesState,START,END


def hello_node(state: MessagesState): 
    # Return a state update: add one assistant message 
    return {"messages": [{"role": "ai", "content": "Hello World"}]}
builder = StateGraph(MessagesState) 
builder.add_node("hello", hello_node) 
builder.add_edge(START, "hello") 
builder.add_edge("hello", END)
graph = builder.compile() 
result = graph.invoke({"messages": [{"role": "user", "content": "Hi"}]}) 
print(result["messages"][-1].content)
