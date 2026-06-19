import asyncio 
from langchain.agents import create_agent 
from langchain_mcp_adapters.client import MultiServerMCPClient 
from langchain.agents import create_agent 
from langchain.messages import HumanMessage 
from langgraph.checkpoint.memory import InMemorySaver 
from dotenv import load_dotenv

async def main(): 
    client = MultiServerMCPClient( 
        { 
            "travel_server": { 
                "transport": "streamable_http", 
                "url": "https://mcp.kiwi.com" 
                } 
        } )
    tools = await client.get_tools() 
    load_dotenv()
    agent = create_agent( "gpt-5-nano", 
                         tools=tools,
                          checkpointer=InMemorySaver(), 
                         system_prompt="You are a travel agent. No follow up questions." ) 
    config = {"configurable": {"thread_id": "1"}}
    response = await agent.ainvoke( {"messages": [HumanMessage(content="Get me a direct flight from Rabat to Agadir on August 31st")]}, config ) 
    print(response)
asyncio.run(main())