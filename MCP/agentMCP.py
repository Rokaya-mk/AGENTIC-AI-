import asyncio
from langchain.agents import create_agent 
from langchain_mcp_adapters.client import MultiServerMCPClient 
from langchain.messages import HumanMessage 
from langchain_ollama import ChatOllama

async def main():
    client = MultiServerMCPClient(
        {
            "local_server": {
                "transport": "stdio",
                "command": "python",
                "args": ["resources/mcp_local_server.py"],
            }
        }
    )
    tools = await client.get_tools()
    resources = await client.get_resources("local_server")
    prompt = await client.get_prompt("local_server", "prompt")
    prompt = prompt[0].content

    model = ChatOllama(
        model="llama3.2:latest",
        temperature=0,
    )
    agent = create_agent(
        model=model, 
        tools=tools, 
        system_prompt=prompt
        )
    config = {"configurable": {"thread_id": "1"}}
    response = await agent.ainvoke(
        {"messages": [HumanMessage(content="Tell me about the langchain-mcp-adapters library")]},
        config=config,
    )
    print(response)

asyncio.run(main())
