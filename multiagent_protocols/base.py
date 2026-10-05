from __future__ import annotations

from dataclasses import asdict, dataclass, field
import importlib
import re
from typing import Any


_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass(frozen=True)
class ProtocolPaths:
    comm_dir: str
    shared_prompt: str

    @classmethod
    def default(cls, comm_dir: str = "/tmp/multiagent-collab") -> "ProtocolPaths":
        return cls(
            comm_dir=comm_dir,
            shared_prompt=f"{comm_dir}/shared-prompt.md",
        )


@dataclass(frozen=True)
class AgentSpec:
    id: str
    prompt: str


@dataclass(frozen=True)
class ProtocolSpec:
    agents: list[AgentSpec]
    shared_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


def build_protocol(
    import_path: str,
    instruction: str,
    paths: ProtocolPaths,
    model: str,
) -> ProtocolSpec:

    module, _, attribute = import_path.partition(":")
    protocol = getattr(importlib.import_module(module), attribute or "MultiAgentProtocol")()
    spec = protocol.build(instruction=instruction, paths=paths, model=model)
    if not isinstance(spec, ProtocolSpec):
        raise TypeError("Protocol build() must return ProtocolSpec")
    validate_protocol_spec(spec)
    return spec


def validate_protocol_spec(spec: ProtocolSpec) -> None:
    if not spec.agents:
        raise ValueError("ProtocolSpec must define at least one agent")
    seen: set[str] = set()
    for agent in spec.agents:
        if not isinstance(agent, AgentSpec):
            raise TypeError("ProtocolSpec.agents must contain AgentSpec values")
        if not _SAFE_ID.match(agent.id):
            raise ValueError(f"Unsafe agent id: {agent.id!r}")
        if agent.id in seen:
            raise ValueError(f"Duplicate agent id: {agent.id!r}")
        if not agent.prompt.strip():
            raise ValueError(f"Agent {agent.id!r} has an empty prompt")
        seen.add(agent.id)


def protocol_to_dict(spec: ProtocolSpec) -> dict[str, Any]:
    return asdict(spec)
