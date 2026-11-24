"""Outbox: каждое сообщение сначала на диск (SQLite), удаляем только после PUBACK брокера.

Зачем: paho держит неподтверждённые сообщения только в памяти — при падении процесса
они теряются. Outbox переживает перезапуск: при старте всё, что осталось, уходит снова.

- payload хранится байтами: повтор = те же байты (BE дедуплицирует по хэшу, ключ — event_id);
- key уникален (event_id или event_id:track_id): одно и то же сообщение дважды не встанет;
- порядок отправки = порядок записи (event раньше своего participant);
- срок жизни: после долгой недоступности брокера старое сообщение не отправляется (тревога
  о событии часовой давности вредна) — expire() удаляет его и возвращает для локального журнала.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class OutboxItem:
    id: int
    topic: str
    payload: bytes
    key: str
    attempts: int
    created_at: float = 0.0


class Outbox:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._lock = threading.Lock()
        with self._lock:
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("""CREATE TABLE IF NOT EXISTS outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                payload BLOB NOT NULL,
                key TEXT NOT NULL UNIQUE,
                created_at REAL NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0)""")

    def add(self, topic: str, payload: bytes, key: str) -> bool:
        """True — записано; False — такой key уже есть (дубль не ставим)."""
        with self._lock:
            cur = self._db.execute(
                "INSERT OR IGNORE INTO outbox(topic, payload, key, created_at) VALUES (?, ?, ?, ?)",
                (topic, payload, key, time.time()))
            return cur.rowcount == 1

    def pending(self, limit: int = 100, exclude: set[int] | None = None) -> list[OutboxItem]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, topic, payload, key, attempts, created_at FROM outbox ORDER BY id LIMIT ?",
                (limit + len(exclude or ()),)).fetchall()
        items = [OutboxItem(r[0], r[1], bytes(r[2]), r[3], r[4], r[5]) for r in rows]
        return [i for i in items if not exclude or i.id not in exclude][:limit]

    def expire(self, max_age_s: float, now: float | None = None, keep: set[int] | None = None) -> list[OutboxItem]:
        """Удалить сообщения старше max_age_s (кроме keep — уже у брокера в пути). -> удалённые."""
        cutoff = (now if now is not None else time.time()) - max_age_s
        with self._lock:
            rows = self._db.execute(
                "SELECT id, topic, payload, key, attempts, created_at FROM outbox WHERE created_at < ? ORDER BY id",
                (cutoff,)).fetchall()
            old = [OutboxItem(r[0], r[1], bytes(r[2]), r[3], r[4], r[5]) for r in rows if r[0] not in (keep or ())]
            self._db.executemany("DELETE FROM outbox WHERE id = ?", [(i.id,) for i in old])
        return old

    def mark_attempt(self, item_id: int) -> None:
        with self._lock:
            self._db.execute("UPDATE outbox SET attempts = attempts + 1 WHERE id = ?", (item_id,))

    def delete(self, item_id: int) -> None:
        with self._lock:
            self._db.execute("DELETE FROM outbox WHERE id = ?", (item_id,))

    def __len__(self) -> int:
        with self._lock:
            return self._db.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._db.close()
