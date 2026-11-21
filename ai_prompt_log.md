# AI Engineering Audit & Verification Log

## 1. Initial Scaffolding Prompt & AI Output Summary
> Prompt: Build a production-grade LangGraph ReAct agent in Python 3.11+ with AgentState, MemorySaver checkpointing, SHA-256 cycle bounds, and HITL interrupt_before approval gates.
> AI Draft Output: Generated baseline state.py, tools.py, circuit_breaker.py, graph.py, and app.py.

## 2. Technical Audit Findings & Remediation Diffs

### Flaw 1: In-Place State Mutation Breaking Checkpointer Snapshots
- **Audit Finding:** The AI draft mutated `state["messages"].append(response)` in place, bypassing LangGraph reducers (`add_messages`) and corrupting checkpoint snapshots across `interrupt_before`.
- **Remediation:** Refactored all nodes to return pure partial state dictionaries.
```diff
- state["messages"].append(response)
- state["iteration_count"] += 1
- return state
+ return {"messages": [response], "iteration_count": next_iter}
```

### Flaw 2: Infinite Approval Loop on Human Rejection
- **Audit Finding:** On REJECTED decisions, the AI draft routed back to `agent_node` without injecting a matching `ToolMessage`, violating OpenAI tool-call pairing and causing an infinite re-proposal loop.
- **Remediation:** Injected explicit `AUTHORIZATION DENIED` `ToolMessage` entries in `human_approval_gate` and added a system rule forbidding re-attempts.
```diff
- if decision == "REJECTED":
-     return {"approval_status": None}
+ if decision == "REJECTED":
+     return {"messages": rejection_messages, "pending_approval": None, "approval_status": None}
```

### Flaw 3: Unnormalized Dictionary Hashing in Circuit Breaker
- **Audit Finding:** Hashing `str(args)` directly produced different SHA-256 digests when dictionary keys appeared in different orders, allowing duplicate loops to evade detection.
- **Remediation:** Canonicalized argument JSON via `json.dumps({"tool_name": tool_name, "args": args}, sort_keys=True)` before SHA-256 hashing.
```diff
- raw = f"{tool_name}:{str(args)}"
+ canonical_payload = json.dumps({"tool_name": tool_name, "args": args or {}}, sort_keys=True, separators=(",", ":"))
```

### Flaw 4: Hardcoded Global thread_id Causing Cross-Session State Pollution
- **Audit Finding:** The AI draft used a static `thread_id="default"` in Streamlit, causing concurrent sessions and approval tickets to collide in `MemorySaver`.
- **Remediation:** Bound isolated UUID thread IDs (`ops-session-<hex>`) to `st.session_state` with a manual thread switcher and reset button.
```diff
- config = {"configurable": {"thread_id": "default"}}
+ st.session_state.thread_id = f"ops-session-{uuid.uuid4().hex[:6]}"
```
