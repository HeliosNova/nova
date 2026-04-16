"""Heartbeat system — monitors, change detection, and proactive alerting.

Backward-compatibility re-export. The implementation has been split into:
  - monitor_store.py  — Monitor/MonitorResult/HeartbeatInstruction dataclasses,
                        MonitorStore CRUD, and change-detection utilities
  - heartbeat_loop.py — HeartbeatLoop scheduling engine + all monitor runners
"""

from __future__ import annotations

from app.monitors.monitor_store import (
    Monitor,
    MonitorResult,
    HeartbeatInstruction,
    MonitorStore,
    extract_numbers,
    detect_change,
)
from app.monitors.heartbeat_loop import (
    HeartbeatLoop,
    _strip_deliberation,
    _DELIBERATION_PATTERNS,
)

__all__ = [
    # Data types
    "Monitor",
    "MonitorResult",
    "HeartbeatInstruction",
    # Store
    "MonitorStore",
    # Change detection
    "extract_numbers",
    "detect_change",
    # Loop engine
    "HeartbeatLoop",
    # Internals used by tests
    "_strip_deliberation",
    "_DELIBERATION_PATTERNS",
]
