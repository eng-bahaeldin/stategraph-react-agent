import os
from typing import Any, Literal, Optional
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from src.circuit_breaker import MAX_ITERATIONS, evaluate_circuit_breaker
from src.state import AgentState
from src.tools import ALL_TOOLS, TOOL_REGISTRY, get_tool_risk_level, is_tool_sensitive

load_dotenv()

SYSTEM_PROMPT = """You are an enterprise Internal Cloud & Financial Operations ReAct Agent.
Tools available:
- query_service_metrics(service_name): Read-only telemetry check.
- check_billing_balance(account_id): Read-only balance check.
- scale_kubernetes_cluster(service_name, replica_count): Sensitive production scaling mutation.
- execute_wire_transfer(recipient_id, amount_usd): Sensitive outbound financial transfer.

Rules:
1. Gather diagnostic telemetry with read-only tools before proposing sensitive actions.
2. CRITICAL AUTHORIZATION RULE: If a ToolMessage states that a human operator REJECTED authorization for an action, immediately respect the denial, NEVER re-propose or re-invoke that tool, and clearly tell the operator the action was cancelled.
"""


def _get_default_llm() -> BaseChatModel:
    api_key = os.getenv("OPENAI_API_KEY", "sk-placeholder-for-offline-tests")
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)


def build_state_graph(
    llm: Optional[BaseChatModel] = None,
    checkpointer: Optional[MemorySaver] = None,
):
    """Constructs and compiles the deterministic StateGraph with HITL interruption."""
    base_llm = llm or _get_default_llm()
    bound_llm = base_llm.bind_tools(ALL_TOOLS)
    memory = checkpointer if checkpointer is not None else MemorySaver()

    def agent_node(state: AgentState) -> dict[str, Any]:
        next_iter = state.get("iteration_count", 0) + 1
        messages = state.get("messages", [])
        response = bound_llm.invoke([SystemMessage(content=SYSTEM_PROMPT), *messages])
        return {"messages": [response], "iteration_count": next_iter}

    def circuit_breaker_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        tool_calls = getattr(last_msg, "tool_calls", []) or []

        tripped, reason, updated_hashes = evaluate_circuit_breaker(
            iteration_count=state.get("iteration_count", 0),
            tool_calls=tool_calls,
            executed_tool_hashes=state.get("executed_tool_hashes", []),
            max_iterations=MAX_ITERATIONS,
        )
        if tripped:
            return {
                "circuit_broken": True,
                "error_reason": reason,
                "executed_tool_hashes": updated_hashes,
                "pending_approval": None,
            }

        sensitive_calls = [tc for tc in tool_calls if is_tool_sensitive(tc["name"])]
        pending_payload = None
        if sensitive_calls:
            pending_payload = {
                "actions": [
                    {
                        "tool_call_id": tc["id"],
                        "tool_name": tc["name"],
                        "args": tc.get("args", {}),
                        "risk_level": get_tool_risk_level(tc["name"]),
                    }
                    for tc in sensitive_calls
                ]
            }
        return {
            "circuit_broken": False,
            "error_reason": None,
            "executed_tool_hashes": updated_hashes,
            "pending_approval": pending_payload,
            "approval_status": None,
        }


    def human_approval_gate(state: AgentState) -> dict[str, Any]:
        decision = state.get("approval_status")
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        tool_calls = getattr(last_msg, "tool_calls", []) or []

        if decision == "REJECTED":
            rejection_messages = []
            for tc in tool_calls:
                t_name = tc["name"]
                rejection_messages.append(
                    ToolMessage(
                        tool_call_id=tc["id"],
                        name=t_name,
                        content=f"AUTHORIZATION DENIED: Operator REJECTED [{t_name}]. Do NOT re-attempt.",
                    )
                )
            return {"messages": rejection_messages, "pending_approval": None, "approval_status": None}

        return {"pending_approval": None, "approval_status": None}

    def tool_executor_node(state: AgentState) -> dict[str, Any]:
        messages = state.get("messages", [])
        last_msg = messages[-1] if messages else None
        tool_calls = getattr(last_msg, "tool_calls", []) or []
        simulate_failure = state.get("simulate_tool_failure", False)

        tool_outputs: list[ToolMessage] = []
        for tc in tool_calls:
            t_name = tc["name"]
            t_args = tc.get("args", {})
            if simulate_failure and not is_tool_sensitive(t_name):
                output_str = f"TRANSIENT_ERROR (504 Gateway Timeout): [{t_name}] failed. Retry recommended."
            else:
                tool_fn = TOOL_REGISTRY.get(t_name)
                output_str = str(tool_fn.invoke(t_args)) if tool_fn else f"ERROR: Unknown tool [{t_name}]."
            tool_outputs.append(ToolMessage(tool_call_id=tc["id"], name=t_name, content=output_str))
        return {"messages": tool_outputs}

    def error_node(state: AgentState) -> dict[str, Any]:
        reason = state.get("error_reason") or "Execution halted by safety guardrails."
        return {
            "messages": [AIMessage(content=f"[CIRCUIT BREAKER TERMINATION] {reason}")],
            "pending_approval": None,
        }

    def route_after_agent(state: AgentState) -> Literal["circuit_breaker_node", "__end__"]:
        last_msg = state["messages"][-1] if state.get("messages") else None
        return "circuit_breaker_node" if getattr(last_msg, "tool_calls", None) else "__end__"

    def route_after_cb(state: AgentState) -> Literal["error_node", "human_approval_gate", "tool_executor_node"]:
        if state.get("circuit_broken", False):
            return "error_node"
        if state.get("pending_approval") is not None:
            return "human_approval_gate"
        return "tool_executor_node"

    def route_after_approval(state: AgentState) -> Literal["tool_executor_node", "agent_node"]:
        msgs = state.get("messages", [])
        return "agent_node" if (msgs and isinstance(msgs[-1], ToolMessage)) else "tool_executor_node"

    workflow = StateGraph(AgentState)
    workflow.add_node("agent_node", agent_node)
    workflow.add_node("circuit_breaker_node", circuit_breaker_node)
    workflow.add_node("human_approval_gate", human_approval_gate)
    workflow.add_node("tool_executor_node", tool_executor_node)
    workflow.add_node("error_node", error_node)

    workflow.add_edge(START, "agent_node")
    workflow.add_conditional_edges("agent_node", route_after_agent, {"circuit_breaker_node": "circuit_breaker_node", "__end__": END})
    workflow.add_conditional_edges("circuit_breaker_node", route_after_cb, {"error_node": "error_node", "human_approval_gate": "human_approval_gate", "tool_executor_node": "tool_executor_node"})
    workflow.add_conditional_edges("human_approval_gate", route_after_approval, {"tool_executor_node": "tool_executor_node", "agent_node": "agent_node"})
    workflow.add_edge("tool_executor_node", "agent_node")
    workflow.add_edge("error_node", END)

    return workflow.compile(checkpointer=memory, interrupt_before=["human_approval_gate"])
