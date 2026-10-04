"""Linear version history for the daily record (ported from TCM-Meridian, NOTE only)."""
from __future__ import annotations

from datetime import datetime
from uuid import uuid4


class SnapshotHistory:
    def __init__(self):
        self.snapshots: list[dict] = []
        self.current_index = -1

    def push(self, note: str, source: str = 'init', t: float | None = None,
             meta: dict | None = None) -> list[dict]:
        """Append a version after the current one; returns the truncated redo branch for auditing."""
        truncated = self.snapshots[self.current_index + 1:]
        self.snapshots = self.snapshots[:self.current_index + 1]
        self.snapshots.append({'id': str(uuid4()), 'note': note, 'source': source,
                               'timestamp': datetime.now().isoformat(timespec='seconds'),
                               't': t, 'meta': meta or {}})
        self.current_index = len(self.snapshots) - 1
        return truncated

    def can_undo(self) -> bool:
        return self.current_index > 0

    def can_redo(self) -> bool:
        return self.current_index < len(self.snapshots) - 1

    def undo(self) -> dict | None:
        if self.can_undo():
            self.current_index -= 1
            return self.snapshots[self.current_index]
        return None

    def redo(self) -> dict | None:
        if self.can_redo():
            self.current_index += 1
            return self.snapshots[self.current_index]
        return None

    def get_current(self) -> dict | None:
        return self.snapshots[self.current_index] if 0 <= self.current_index < len(self.snapshots) else None

    def get_previous(self) -> dict | None:
        return self.snapshots[self.current_index - 1] if self.current_index > 0 else None

    def current_note(self) -> str:
        current = self.get_current()
        return current['note'] if current else ''

    def to_dict(self) -> dict:
        return {'current_index': self.current_index, 'snapshots': self.snapshots}

    def restore(self, data: dict):
        snapshots = [s for s in data.get('snapshots', []) if isinstance(s, dict)]
        self.snapshots = snapshots
        try:
            index = int(data.get('current_index', len(snapshots) - 1))
        except (TypeError, ValueError):
            index = len(snapshots) - 1
        self.current_index = max(0, min(index, len(snapshots) - 1)) if snapshots else -1
