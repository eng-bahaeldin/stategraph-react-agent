import hashlib
import json
from typing import Any, Optional

MAX_ITERATIONS: int = 6


def compute_tool_call_hash(tool_name: str, args: dict[str, Any]) -> str:
    canonical_payload = json.dumps(
        {"tool_name": tool_name, "args": args or {}},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def evaluate_circuit_breaker(
    iteration_count: int,
    tool_calls: list[dict[str, Any]],
    executed_tool_hashes: list[str],
    max_iterations: int = MAX_ITERATIONS,
) -> tuple[bool, Optional[str], list[str]]:
    if iteration_count >= max_iterations:
        return (
            True,
            f"Hard recursion bound reached (iteration_count={iteration_count} >= {max_iterations}). Execution halted.",
            list(executed_tool_hashes),
        )
    updated_hashes = list(executed_tool_hashes)
    for tc in tool_calls:
        t_name = tc["name"]
        call_hash = compute_tool_call_hash(t_name, tc.get("args", {}))
        if updated_hashes and updated_hashes[-1] == call_hash:
            return (
                True,
                f"Duplicate consecutive tool call detected for [{t_name}] (SHA-256: {call_hash[:12]}...). Circuit breaker tripped to prevent infinite loop.",
                updated_hashes + [call_hash],
            )
        updated_hashes.append(call_hash)
    return False, None, updated_hashes
