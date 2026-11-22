from typing import Any, List, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langgraph.checkpoint.memory import MemorySaver
from src.circuit_breaker import compute_tool_call_hash
from src.graph import build_state_graph


class InfiniteDistinctToolLLM(BaseChatModel):
    counter: int = 0

    @property
    def _llm_type(self) -> str:
        return "infinite-distinct"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.counter += 1
        msg = AIMessage(content="", tool_calls=[{"id": f"call_{self.counter}", "name": "query_service_metrics", "args": {"service_name": f"node-{self.counter}"}}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_maximum_iteration_hard_bound() -> None:
    graph = build_state_graph(llm=InfiniteDistinctToolLLM(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-max-iter"}}
    final_state = graph.invoke({"messages": [HumanMessage(content="Scan all nodes")], "iteration_count": 0}, config=config)
    assert final_state["iteration_count"] == 6
    assert final_state["circuit_broken"] is True
    assert "Hard recursion bound reached" in final_state["error_reason"]
    assert "CIRCUIT BREAKER TERMINATION" in final_state["messages"][-1].content


class RepeatIdenticalToolLLM(BaseChatModel):
    turn: int = 0

    @property
    def _llm_type(self) -> str:
        return "repeat-identical"

    def bind_tools(self, tools: Any, **kwargs: Any) -> Runnable:
        return self

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Any = None, **kwargs: Any) -> ChatResult:
        self.turn += 1
        args = {"service_name": "auth-service", "extra": "1"} if self.turn == 1 else {"extra": "1", "service_name": "auth-service"}
        msg = AIMessage(content="", tool_calls=[{"id": f"dup_{self.turn}", "name": "query_service_metrics", "args": args}])
        return ChatResult(generations=[ChatGeneration(message=msg)])


def test_sha256_payload_hash_tripping() -> None:
    h1 = compute_tool_call_hash("scale_kubernetes_cluster", {"service_name": "api", "replica_count": 5})
    h2 = compute_tool_call_hash("scale_kubernetes_cluster", {"replica_count": 5, "service_name": "api"})
    assert h1 == h2
    graph = build_state_graph(llm=RepeatIdenticalToolLLM(), checkpointer=MemorySaver())
    config = {"configurable": {"thread_id": "test-repeat-hash"}}
    final_state = graph.invoke({"messages": [HumanMessage(content="Check auth-service repeatedly")], "iteration_count": 0}, config=config)
    assert final_state["circuit_broken"] is True
    assert "Duplicate consecutive tool call detected" in final_state["error_reason"]
    assert len(final_state["executed_tool_hashes"]) == 2
    assert final_state["executed_tool_hashes"][0] == final_state["executed_tool_hashes"][1]
