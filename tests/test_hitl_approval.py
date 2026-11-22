from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from src.graph import build_state_graph


class SensitiveActionLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "sensitive-action-llm"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if not tool_msgs:
            msg = AIMessage(content="", tool_calls=[{"id": "call_wire_8812", "name": "execute_wire_transfer", "args": {"recipient_id": "V-8812", "amount_usd": 1500.0}}])
            return ChatResult(generations=[ChatGeneration(message=msg)])
        last_out = str(tool_msgs[-1].content)
        if "AUTHORIZATION DENIED" in last_out:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Understood. The $1,500 wire transfer to V-8812 was cancelled per your rejection."))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Completed: $1,500.00 wire transfer to Vendor V-8812 has been executed."))])


def test_sensitive_action_interruption() -> None:
    graph = build_state_graph(llm=SensitiveActionLLM(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-hitl-interrupt"}}
    graph.invoke({"messages": [HumanMessage(content="Transfer $1,500 to V-8812")], "iteration_count": 0}, config=config)
    snap = graph.get_state(config)
    assert snap.next == ("human_approval_gate",)
    assert snap.values["pending_approval"] is not None
    assert snap.values["pending_approval"]["actions"][0]["tool_name"] == "execute_wire_transfer"


def test_approval_resumption() -> None:
    graph = build_state_graph(llm=SensitiveActionLLM(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-hitl-approve"}}
    graph.invoke({"messages": [HumanMessage(content="Transfer $1,500 to V-8812")], "iteration_count": 0}, config=config)
    graph.update_state(config, {"approval_status": "APPROVED"})
    final_state = graph.invoke(None, config=config)
    snap = graph.get_state(config)
    assert snap.next == ()
    tool_msgs = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "WIRE-2026-8812" in tool_msgs[0].content
    assert "executed" in final_state["messages"][-1].content.lower()


def test_rejection_recovery() -> None:
    graph = build_state_graph(llm=SensitiveActionLLM(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-hitl-reject"}}
    graph.invoke({"messages": [HumanMessage(content="Transfer $1,500 to V-8812")], "iteration_count": 0}, config=config)
    graph.update_state(config, {"approval_status": "REJECTED"})
    final_state = graph.invoke(None, config=config)
    snap = graph.get_state(config)
    assert snap.next == ()
    tool_msgs = [m for m in final_state["messages"] if isinstance(m, ToolMessage)]
    assert len(tool_msgs) == 1
    assert "AUTHORIZATION DENIED" in tool_msgs[0].content
    assert "WIRE-2026-8812" not in tool_msgs[0].content
    assert "cancelled" in final_state["messages"][-1].content.lower()
