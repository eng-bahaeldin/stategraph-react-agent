from langchain_core.tools import BaseTool, tool


@tool
def query_service_metrics(service_name: str) -> str:
    """Safe read-only telemetry lookup for a production microservice."""
    if service_name.strip().lower() == "auth-service":
        return "Service: auth-service | Error Rate: 7.4% (CRITICAL: exceeds 5.0% SLA threshold) | P99 Latency: 840ms | Active Replicas: 3"
    return f"Service: {service_name} | Error Rate: 0.4% (HEALTHY) | P99 Latency: 42ms | Active Replicas: 4"


@tool
def check_billing_balance(account_id: str) -> str:
    """Safe read-only lookup of available operational funds in a corporate billing account."""
    return f"Account ID: {account_id} | Available Balance: $48,250.00 USD | Status: ACTIVE"


@tool
def scale_kubernetes_cluster(service_name: str, replica_count: int) -> str:
    """Sensitive infrastructure mutation: scales production Kubernetes pod replicas."""
    return f"EXECUTED: Scaled production Kubernetes deployment [{service_name}] to {replica_count} replicas."


@tool
def execute_wire_transfer(recipient_id: str, amount_usd: float) -> str:
    """Sensitive financial action: executes an outbound wire transfer to an external vendor."""
    return f"EXECUTED: Wire transfer of ${amount_usd:,.2f} USD completed to Vendor [{recipient_id}]. Reference ID: WIRE-2026-8812."

query_service_metrics.metadata = {"is_sensitive": False, "risk_level": "LOW (Read-Only Telemetry)"}
check_billing_balance.metadata = {"is_sensitive": False, "risk_level": "LOW (Read-Only Billing)"}
scale_kubernetes_cluster.metadata = {"is_sensitive": True, "risk_level": "HIGH (Production Infrastructure Mutation)"}
execute_wire_transfer.metadata = {"is_sensitive": True, "risk_level": "CRITICAL (Outbound Financial Transfer)"}

ALL_TOOLS: list[BaseTool] = [
    query_service_metrics,
    check_billing_balance,
    scale_kubernetes_cluster,
    execute_wire_transfer,
]

TOOL_REGISTRY: dict[str, BaseTool] = {t.name: t for t in ALL_TOOLS}


def is_tool_sensitive(tool_name: str) -> bool:
    tool_obj = TOOL_REGISTRY.get(tool_name)
    if tool_obj is None:
        return True
    meta = tool_obj.metadata or {}
    return bool(meta.get("is_sensitive", True))


def get_tool_risk_level(tool_name: str) -> str:
    tool_obj = TOOL_REGISTRY.get(tool_name)
    if tool_obj is None:
        return "CRITICAL (Unregistered Tool)"
    meta = tool_obj.metadata or {}
    return str(meta.get("risk_level", "HIGH"))
