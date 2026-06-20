from langchain.tools import tool 
from langchain.agents import create_agent 
from langchain.messages import HumanMessage
@tool 
def square_root(x: float) -> float: 
    """Calculate the square root of a number""" 
    return x ** 0.5 
@tool 
def square(x: float) -> float: 
    """Calculate the square of a number""" 
    return x ** 2


subagent_1 = create_agent( model='gpt-5-nano', tools=[square_root] ) 
subagent_2 = create_agent( model='gpt-5-nano', tools=[square] )

@tool 
def call_subagent_1(x: float) -> float: 
    """Call subagent 1 in order to calculate the square root of a number""" 
    response = subagent_1.invoke({"messages": [
        HumanMessage(content=f"Calcu-late the square root of {x}")]}) 
    return response["messages"][-1].content 
@tool 
def call_subagent_2(x: float) -> float: 
    """Call subagent 2 in order to calculate the square of a number""" 
    response = subagent_2.invoke({"messages": [
        HumanMessage(content=f"Calcu-late the square of {x}")]}) 
    return response["messages"][-1].content 
main_agent = create_agent( 
    model='gpt-5-nano', 
    tools=[call_subagent_1, call_subagent_2], 
    system_prompt="You are a helpful assistant who can call subagents to cal-culate the square root or square of a number."
    )
