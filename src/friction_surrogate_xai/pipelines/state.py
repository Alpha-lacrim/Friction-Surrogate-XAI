"""Persistent state for resumable pipeline execution."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from friction_surrogate_xai.eda.utils import ensure_directory


def utc_now() -> str:
    """Return an ISO timestamp in UTC."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StageState:
    """State for one pipeline stage."""

    name: str
    status: str
    started_at: str | None = None
    ended_at: str | None = None
    message: str = ""
    artifact_paths: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "StageState":
        """Build stage state from JSON values."""
        return cls(
            name=str(values.get("name", "")),
            status=str(values.get("status", "pending")),
            started_at=values.get("started_at"),
            ended_at=values.get("ended_at"),
            message=str(values.get("message", "")),
            artifact_paths=list(values.get("artifact_paths", [])),
            metadata=dict(values.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable state."""
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "message": self.message,
            "artifact_paths": self.artifact_paths,
            "metadata": self.metadata,
        }


class PipelineStateStore:
    """Read and write pipeline state after every stage."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.state = self._load()

    def get(self, stage_name: str) -> StageState | None:
        """Return state for a stage if present."""
        values = self.state.get("stages", {}).get(stage_name)
        return StageState.from_dict(values) if values else None

    def mark_running(self, stage_name: str) -> None:
        """Persist a running stage state."""
        self._set(
            StageState(
                name=stage_name,
                status="running",
                started_at=utc_now(),
            )
        )

    def mark_completed(
        self,
        stage_name: str,
        *,
        message: str = "",
        artifact_paths: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Persist a completed stage state."""
        current = self.get(stage_name)
        self._set(
            StageState(
                name=stage_name,
                status="completed",
                started_at=current.started_at if current else utc_now(),
                ended_at=utc_now(),
                message=message,
                artifact_paths=artifact_paths or [],
                metadata=metadata or {},
            )
        )

    def mark_failed(self, stage_name: str, *, message: str) -> None:
        """Persist a failed stage state."""
        current = self.get(stage_name)
        self._set(
            StageState(
                name=stage_name,
                status="failed",
                started_at=current.started_at if current else utc_now(),
                ended_at=utc_now(),
                message=message,
            )
        )

    def mark_skipped(self, stage_name: str, *, message: str) -> None:
        """Persist a skipped stage state."""
        self._set(
            StageState(
                name=stage_name,
                status="skipped",
                started_at=utc_now(),
                ended_at=utc_now(),
                message=message,
            )
        )

    def stage_table(self) -> list[dict[str, Any]]:
        """Return stage states in insertion order."""
        return list(self.state.get("stages", {}).values())

    def _set(self, stage_state: StageState) -> None:
        self.state.setdefault("stages", {})[stage_state.name] = stage_state.to_dict()
        self.state["updated_at"] = utc_now()
        self.save()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"created_at": utc_now(), "updated_at": utc_now(), "stages": {}}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self) -> None:
        """Write state to disk."""
        ensure_directory(self.path.parent)
        self.path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")
