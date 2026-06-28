from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    question: str
    documents: list[dict]
    web_results: list[dict]
    needs_web_search: bool
    generation: str
    retry_count: int
    answer_is_grounded: bool
    answer_addresses_question: bool