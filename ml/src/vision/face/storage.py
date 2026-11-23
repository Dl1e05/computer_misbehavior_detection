"""Хранилище снимков лиц: ML только ЗАПИСЫВАЕТ и отдаёт ключ (snapshot_ref).

Чтения/списка/удаления у ML нет намеренно: смотреть снимки может только BE по ключу.
Сейчас — локальная папка; MinIO (S3, ключ с правом только на запись) — когда BE выберет.
Ключ не публичная ссылка: faces/<camera_id>/<YYYYMMDD>/<event_id>_<track_id>.jpg
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Protocol


class SnapshotStorage(Protocol):
    def put(self, key: str, data: bytes) -> str: ...


def snapshot_key(camera_id: str, event_id: str, track_id: int, ts: datetime) -> str:
    return f"faces/{camera_id}/{ts:%Y%m%d}/{event_id}_{track_id}.jpg"


class LocalSnapshotStorage:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def put(self, key: str, data: bytes) -> str:
        path = (self.root / key).resolve()
        if self.root.resolve() not in path.parents:
            raise ValueError(f"ключ вне хранилища: {key}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key
