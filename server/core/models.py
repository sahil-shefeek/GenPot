# server/core/models.py
"""
Unified data models for protocol-agnostic communication between
network adapters (HTTP, SMTP, …) and the GenPotEngine.
"""

from dataclasses import dataclass, field
from typing import Dict, Union


@dataclass
class UnifiedRequest:
    """Normalised inbound request from any network adapter."""

    protocol: str
    source_ip: str
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: str = ""


@dataclass
class UnifiedResponse:
    """Normalised outbound response returned by the engine."""

    status_code: int
    headers: Dict[str, str] = field(default_factory=dict)
    data: Union[dict, str] = field(default_factory=dict)
