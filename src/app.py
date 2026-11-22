import ast
import json
import sys
import uuid
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from src.circuit_breaker import MAX_ITERATIONS
from src.graph import build_state_graph

st.set_page_config(page_title="OpsGuard ReAct StateGraph", layout="wide")

def _extract_text(content) -> str:
    if isinstance(content, str):
        stripped = content.strip()
        if stripped.startswith("["):
            try:
                content = ast.literal_eval(stripped)
            except Exception:
                return content
        else:
            return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return chr(10).join(p for p in parts if p).strip()
    return str(content) if content else ""

if "checkpointer" not in st.session_state:
    st.session_state.checkpointer = MemorySaver()
    st.session_state.graph = build_state_graph(checkpointer=st.session_state.checkpointer)
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"ops-session-{uuid.uuid4().hex[:6]}"
if "node_trace" not in st.session_state:
    st.session_state.node_trace = ["START"]

graph = st.session_state.graph

with st.sidebar:
    st.header("State Machine Controls")
    tid_in = st.text_input("Session Thread ID:", value=st.session_state.thread_id)
    if tid_in.strip() and tid_in.strip() != st.session_state.thread_id:
        st.session_state.thread_id = tid_in.strip()
        st.session_state.node_trace = ["START"]
        st.rerun()
    if st.button("New Isolated Session", use_container_width=True):
        st.session_state.thread_id = f"ops-session-{uuid.uuid4().hex[:6]}"
        st.session_state.node_trace = ["START"]
        st.rerun()
    st.divider()
    sim_fail = st.toggle("Inject Failing Read-Only Tool (Circuit Breaker Test)", value=False)
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    snap = graph.get_state(config)
    vals = snap.values if snap and snap.values else {}
    next_nodes = list(snap.next) if snap and snap.next else []
    st.divider()
    st.subheader("Execution Telemetry")
    iters = vals.get("iteration_count", 0)
    st.metric("Iteration Depth", f"{iters} / {MAX_ITERATIONS}")
    if "human_approval_gate" in next_nodes:
        status_lbl = "PAUSED AT HUMAN_APPROVAL_GATE"
    elif vals.get("circuit_broken"):
        status_lbl = "HALTED BY CIRCUIT_BREAKER"
    elif iters > 0:
        status_lbl = "COMPLETED (END)"
    else:
        status_lbl = "IDLE (READY)"
    st.info(f"Status: {status_lbl}")
    with st.expander("Live Checkpointer State (JSON)", expanded=False):
        st.json({
            "thread_id": st.session_state.thread_id,
            "next_nodes": next_nodes,
            "iteration_count": iters,
            "circuit_broken": vals.get("circuit_broken", False),
            "error_reason": vals.get("error_reason"),
            "executed_tool_hashes": vals.get("executed_tool_hashes", []),
            "pending_approval": vals.get("pending_approval"),
            "message_count": len(vals.get("messages", [])),
        })

st.title("Internal Cloud & Financial Operations ReAct Agent")
st.caption("Deterministic LangGraph State Machine with Short-Term Memory, SHA-256 Circuit Breaker & HITL Gate")
trace_parts = [f"[{n}]" for n in st.session_state.node_trace]
if "human_approval_gate" in next_nodes:
    trace_parts.append("[PAUSED: HUMAN_APPROVAL_GATE]")
st.code("Active Node Path: " + " -> ".join(trace_parts), language="text")

c1, c2, c3 = st.columns(3)
preset = ""
with c1:
    if st.button("1. Safe Telemetry Check", use_container_width=True):
        preset = "Check the current error rate on the auth-service."
with c2:
    if st.button("2. Incident Remediation + Wire (HITL)", use_container_width=True):
        preset = "Check the current error rate on the auth-service. If over 5%, scale replicas to 10 and transfer $1,500 to Vendor ID V-8812 for emergency mitigation."
with c3:
    if st.button("3. Follow-Up Memory Query", use_container_width=True):
        preset = "What was the exact error rate you just found for auth-service?"

tool_logs = []
for msg in vals.get("messages", []):
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(_extract_text(msg.content))
    elif isinstance(msg, ToolMessage):
        tool_logs.append(f"**{msg.name}**: {_extract_text(msg.content)}")
    elif isinstance(msg, AIMessage):
        txt = _extract_text(msg.content)
        for tc in getattr(msg, "tool_calls", None) or []:
            t_name = tc.get("name", "")
            t_args = json.dumps(tc.get("args", {}))
            tool_logs.append(f"Called `{t_name}({t_args})`")
        if txt:
            with st.chat_message("assistant"):
                if tool_logs:
                    with st.expander(f"Tool Execution Log ({len(tool_logs)} steps)", expanded=False):
                        for entry in tool_logs:
                            st.markdown(f"- {entry}")
                    tool_logs = []
                st.markdown(txt)
if tool_logs:
    with st.chat_message("assistant"):
        with st.expander(f"Tool Execution Log ({len(tool_logs)} steps)", expanded=True):
            for entry in tool_logs:
                st.markdown(f"- {entry}")

if "human_approval_gate" in next_nodes:
    pending = vals.get("pending_approval") or {}
    with st.container(border=True):
        st.warning("HUMAN-IN-THE-LOOP (HITL) AUTHORIZATION REQUIRED")
        for act in pending.get("actions", []):
            a_name = act.get("tool_name", "")
            a_args = json.dumps(act.get("args", {}))
            a_risk = act.get("risk_level", "HIGH")
            st.markdown(f"- **Action:** `{a_name}` | **Args:** `{a_args}` | **Risk:** **{a_risk}**")
        b_app, b_rej = st.columns(2)
        with b_app:
            if st.button("Approve Action", type="primary", use_container_width=True):
                graph.update_state(config, {"approval_status": "APPROVED"})
                for ev in graph.stream(None, config=config, stream_mode="updates"):
                    st.session_state.node_trace.extend(list(ev.keys()))
                st.rerun()
        with b_rej:
            if st.button("Reject Action", type="secondary", use_container_width=True):
                graph.update_state(config, {"approval_status": "REJECTED"})
                for ev in graph.stream(None, config=config, stream_mode="updates"):
                    st.session_state.node_trace.extend(list(ev.keys()))
                st.rerun()

user_query = st.chat_input("Enter operator command...") or preset
if user_query and "human_approval_gate" not in next_nodes:
    payload = {"messages": [HumanMessage(content=user_query)], "iteration_count": 0, "circuit_broken": False, "error_reason": None, "simulate_tool_failure": sim_fail}
    if not vals:
        payload.update({"executed_tool_hashes": [], "pending_approval": None, "approval_status": None})
    st.session_state.node_trace = ["START"]
    try:
        with st.spinner("Running StateGraph..."):
            for ev in graph.stream(payload, config=config, stream_mode="updates"):
                st.session_state.node_trace.extend(list(ev.keys()))
        st.rerun()
    except Exception as exc:
        st.error(f"Graph execution error: {exc}")
