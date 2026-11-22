from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from src.graph import build_state_graph


class ScriptedChatModel(BaseChatModel):
    responses: List[AIMessage]
    call_index: int = 0

    @property
    def _llm_type(self) -> str:
        return "scripted-mock"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        idx = min(self.call_index, len(self.responses) - 1)
        self.call_index += 1
        return ChatResult(generations=[ChatGeneration(message=self.responses[idx])])


def test_safe_tool_autonomous_traversal() -> None:
    mock_llm = ScriptedChatModel(
        responses=[
            AIMessage(content="", tool_calls=[{"id": "call_safe_1", "name": "query_service_metrics", "args": {"service_name": "auth-service"}}]),
            AIMessage(content="The auth-service error rate is currently 7.4%."),
        ]
    )
    graph = build_state_graph(llm=mock_llm, checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-safe-flow"}}
    visited = []
    for ev in graph.stream({"messages": [HumanMessage(content="Check auth-service")]}, config=config, stream_mode="updates"):
        visited.extend(list(ev.keys()))
    assert visited == ["agent_node", "circuit_breaker_node", "tool_executor_node", "agent_node"]
    snap = graph.get_state(config)
    assert snap.next == ()
    assert "7.4%" in snap.values["messages"][-1].content


class MemoryEchoLLM(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "memory-echo"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        human_texts = [str(m.content) for m in messages if isinstance(m, HumanMessage)]
        if len(human_texts) == 1:
            reply = "Nice to meet you, Alice!"
        else:
            prior = " | ".join(human_texts[:-1])
            reply = f"Earlier you said: {prior}. Your name is Alice."
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])


def test_multi_turn_memory_checkpointing() -> None:
    graph = build_state_graph(llm=MemoryEchoLLM(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-memory-thread"}}
    graph.invoke({"messages": [HumanMessage(content="My name is Alice")]}, config=config)
    res = graph.invoke({"messages": [HumanMessage(content="What is my name?")], "iteration_count": 0}, config=config)
    assert "Alice" in res["messages"][-1].content
    assert len([m for m in res["messages"] if isinstance(m, HumanMessage)]) == 2
