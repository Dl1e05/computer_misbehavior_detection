"""Статус узла ML: камеры (online/offline) и правила (включено / выключено и почему).

Из этого в шаге 8 собирается heartbeat для BE (раз в 5 с): BE видит, что, например,
на cam-02 правило fire выключено из-за отсутствия весов, а не «молчит».
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class NodeStatus:
    node_id: str = "ml-node-1"
    cameras: dict[str, str] = field(default_factory=dict)              # camera_id -> online/offline
    rules: dict[str, dict[str, str]] = field(default_factory=dict)     # camera_id -> {rule: "on" | "off: ..."}
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_camera(self, camera_id: str, status: str) -> None:
        with self._lock:
            self.cameras[camera_id] = status

    def set_rule(self, camera_id: str, rule: str, state: str) -> None:
        with self._lock:
            self.rules.setdefault(camera_id, {})[rule] = state

    def snapshot(self) -> dict:
        with self._lock:
            return {"node_id": self.node_id,
                    "cameras": [{"camera_id": c, "status": s, "rules": dict(self.rules.get(c, {}))}
                                for c, s in sorted(self.cameras.items())]}

    def describe(self) -> str:
        lines = []
        for cam in self.snapshot()["cameras"]:
            rules = ", ".join(f"{r}: {s}" for r, s in cam["rules"].items()) or "нет правил"
            lines.append(f"  {cam['camera_id']} [{cam['status']}] — {rules}")
        return "\n".join(lines)
