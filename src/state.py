from typing import Annotated, Any, Literal, Optional, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Deterministic state schema passed across nodes and persisted by MemorySaver."""

    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int
    executed_tool_hashes: list[str]
    pending_approval: Optional[dict[str, Any]]
    approval_status: Optional[Literal["APPROVED", "REJECTED"]]
    circuit_broken: bool
    error_reason: Optional[str]
    simulate_tool_failure: bool
