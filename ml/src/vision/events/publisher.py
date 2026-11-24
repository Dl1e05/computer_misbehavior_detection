"""Publisher (стадия 4): outbox -> MQTT QoS 1, heartbeat раз в heartbeat_s, Last Will.

Как это работает:
- Analytics вызывает submit_event / submit_participant: сообщение сериализуется ОДИН раз
  и пишется в outbox (SQLite). Больше ничего — конвейер не ждёт сеть.
- Свой поток берёт из outbox по порядку и отдаёт paho (QoS 1, retain=False).
  Строка удаляется только в on_publish, то есть после PUBACK брокера.
- Брокер недоступен -> сообщения копятся в outbox; переподключение 1, 2, 4… до 30 с;
  после подключения всё неподтверждённое отправляется снова теми же байтами.
  Процесс упал -> при следующем старте outbox дошлёт остатки.
- Last Will: брокер сам публикует {"status": "offline"} в heartbeat-топик, если процесс
  пропал без DISCONNECT. При штатной остановке шлём offline-heartbeat сами.
- Heartbeat не идёт через outbox: старый статус после перезапуска никому не нужен;
  QoS 0 и без retain (решение BE 25.11) — старые heartbeat не копятся. Last Will и offline
  при остановке — QoS 1: одно сообщение, терять нельзя.
- Срок жизни (max_age_s): событие/участник старше — не отправляется (после долгой недоступности
  брокера тревога о старом событии вредна), а пишется в локальный журнал dropped_log
  и считается в heartbeat.dropped_stale.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from datetime import datetime, timezone
from typing import Callable

from vision.events.outbox import Outbox
from vision.events.schema import (CameraStatus, Event, Heartbeat, Participant, topic_event, topic_heartbeat,
                                  topic_participant)

log = logging.getLogger("publisher")


def _default_client(client_id: str):
    import paho.mqtt.client as mqtt

    return mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=client_id,
                       protocol=mqtt.MQTTv311, clean_session=True)


class Publisher:
    def __init__(self, node_id: str, run_id: str, outbox: Outbox, host: str = "localhost", port: int = 1883,
                 username: str | None = None, password: str | None = None, keepalive: int = 10,
                 heartbeat_s: float = 5.0, status_fn: Callable[[], dict] | None = None,
                 model_version: str | None = None, max_inflight: int = 20,
                 client_factory: Callable[[str], object] = _default_client, tls: bool = False,
                 max_age_s: float = 600.0, dropped_log: str | Path | None = None):
        self.node_id, self.run_id = node_id, run_id
        self.outbox = outbox
        self.host, self.port, self.keepalive = host, port, keepalive
        self.heartbeat_s = heartbeat_s
        self.status_fn = status_fn
        self.model_version = model_version
        self.max_inflight = max_inflight
        self.max_age_s = max_age_s
        self.dropped_log = Path(dropped_log) if dropped_log else None
        self.dropped_stale = 0
        self._next_expire = 0.0
        self.hb_topic = topic_heartbeat(node_id)
        self.client = client_factory(f"{node_id}-{run_id}")
        if username:
            self.client.username_pw_set(username, password)
        if tls:
            self.client.tls_set()
        self.client.will_set(self.hb_topic, self._heartbeat("offline", with_time=False), qos=1, retain=False)
        self.client.reconnect_delay_set(min_delay=1, max_delay=30)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_publish = self._on_publish

        self._lock = threading.Lock()
        self._inflight: dict[int, int] = {}       # mid -> id строки outbox
        self._early_acks: set[int] = set()        # PUBACK пришёл раньше, чем мы запомнили mid
        self._connected = threading.Event()
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None
        self.sent = 0                             # подтверждено брокером
        self.heartbeats = 0

    @classmethod
    def from_config(cls, cfg: dict, run_id: str, outbox: Outbox, **kw) -> "Publisher":
        return cls(cfg.get("node_id", "ml-node-1"), run_id, outbox,
                   host=cfg.get("host", "localhost"), port=int(cfg.get("port", 1883)),
                   username=os.getenv(cfg.get("user_env", "MQTT_USER")) or None,
                   password=os.getenv(cfg.get("password_env", "MQTT_PASS")) or None,
                   keepalive=int(cfg.get("keepalive_s", 10)), heartbeat_s=float(cfg.get("heartbeat_s", 5)),
                   tls=bool(cfg.get("tls", False)), max_age_s=float(cfg.get("max_age_s", 600)),
                   dropped_log=cfg.get("dropped_log"), **kw)

    # ---------- для конвейера ----------

    def submit_event(self, event: Event) -> bool:
        ok = self.outbox.add(topic_event(event.camera_id), event.to_bytes(), str(event.event_id))
        self._wake.set()
        return ok

    def submit_participant(self, p: Participant) -> bool:
        ok = self.outbox.add(topic_participant(p.camera_id), p.to_bytes(), f"{p.event_id}:{p.track_id}")
        self._wake.set()
        return ok

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def start(self) -> "Publisher":
        self.client.connect_async(self.host, self.port, keepalive=self.keepalive)
        self.client.loop_start()
        self._thread = threading.Thread(target=self._run, name="publisher", daemon=True)
        self._thread.start()
        return self

    def stop(self, flush_timeout: float = 3.0) -> None:
        """Дождаться отправки (до flush_timeout), offline-heartbeat, отключиться."""
        deadline = time.monotonic() + flush_timeout
        while len(self.outbox) and self.connected and time.monotonic() < deadline:
            self._wake.set()
            time.sleep(0.05)
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self.connected:
            info = self.client.publish(self.hb_topic, self._heartbeat("offline"), qos=1, retain=False)
            try:
                info.wait_for_publish(timeout=2)
            except Exception:  # noqa: BLE001 — при остановке не важно
                pass
        self.client.disconnect()
        self.client.loop_stop()

    # ---------- поток отправки ----------

    def _run(self) -> None:
        next_hb = 0.0
        while not self._stop.is_set():
            self._wake.wait(timeout=0.2)
            self._wake.clear()
            self._expire()
            if not self.connected:
                continue
            now = time.monotonic()
            if now >= next_hb:
                self._send_heartbeat()
                next_hb = now + self.heartbeat_s
            self._drain()

    def _expire(self) -> None:
        """Раз в секунду: выбросить из outbox то, что старше max_age_s (не трогая уже отправленное)."""
        if self.max_age_s <= 0 or time.monotonic() < self._next_expire:
            return
        self._next_expire = time.monotonic() + 1.0
        with self._lock:
            busy = set(self._inflight.values())
        old = self.outbox.expire(self.max_age_s, keep=busy)
        if not old:
            return
        self.dropped_stale += len(old)
        log.warning("MQTT: %d сообщений старше %.0f с не отправлены (брокер был недоступен) -> %s",
                    len(old), self.max_age_s, self.dropped_log or "только счётчик")
        if self.dropped_log:
            self.dropped_log.parent.mkdir(parents=True, exist_ok=True)
            with self.dropped_log.open("a", encoding="utf-8") as fh:
                for it in old:
                    fh.write(json.dumps({"dropped_at": time.time(), "created_at": it.created_at, "topic": it.topic,
                                         "key": it.key, "payload": it.payload.decode("utf-8", "replace")},
                                        ensure_ascii=False) + "\n")

    def _drain(self) -> None:
        with self._lock:
            busy = set(self._inflight.values())
        room = self.max_inflight - len(busy)
        if room <= 0:
            return
        for item in self.outbox.pending(limit=room, exclude=busy):
            if not self.connected:
                return
            info = self.client.publish(item.topic, item.payload, qos=1, retain=False)
            if info.rc != 0:                       # не подключены — попробуем после переподключения
                return
            self.outbox.mark_attempt(item.id)
            with self._lock:
                if info.mid in self._early_acks:   # брокер уже подтвердил
                    self._early_acks.discard(info.mid)
                    self._done(item.id)
                else:
                    self._inflight[info.mid] = item.id

    def _done(self, item_id: int) -> None:
        self.outbox.delete(item_id)
        self.sent += 1

    def _heartbeat(self, status: str, with_time: bool = True) -> bytes:
        snap = self.status_fn() if (self.status_fn and status == "online") else {}
        hb = Heartbeat(node_id=self.node_id, run_id=self.run_id, status=status,
                       sent_at=datetime.now(timezone.utc) if with_time else None,
                       model_version=self.model_version,
                       cameras=[CameraStatus(**c) for c in snap.get("cameras", [])],
                       dropped_stale=self.dropped_stale if status == "online" else None)
        return hb.to_bytes()

    def _send_heartbeat(self) -> None:
        try:
            payload = self._heartbeat("online")
        except ValueError as e:                    # статус не по контракту — не роняем поток отправки
            log.error("heartbeat не по контракту: %s", e)
            return
        info = self.client.publish(self.hb_topic, payload, qos=0, retain=False)   # QoS 0: старые не копятся
        if info.rc == 0:
            self.heartbeats += 1

    # ---------- колбэки paho (поток сети) ----------

    def _on_connect(self, client, userdata, flags, reason_code, properties=None):
        if not getattr(reason_code, "is_failure", reason_code != 0):
            log.info("MQTT: подключено к %s:%s, в outbox %d", self.host, self.port, len(self.outbox))
            with self._lock:
                self._inflight.clear()             # всё неподтверждённое — отправить снова из outbox
                self._early_acks.clear()
            self._connected.set()
            self._wake.set()
        else:
            log.warning("MQTT: отказ подключения: %s", reason_code)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties=None):
        if self.connected and not self._stop.is_set():
            log.warning("MQTT: отключено (%s), события копятся в outbox", reason_code)
        self._connected.clear()

    def _on_publish(self, client, userdata, mid, reason_code=None, properties=None):
        # вызывается под внутренним мьютексом paho — здесь не ждём и не публикуем
        with self._lock:
            item_id = self._inflight.pop(mid, None)
            if item_id is None:
                self._early_acks.add(mid)          # heartbeat или PUBACK раньше записи mid
                if len(self._early_acks) > 10_000:
                    self._early_acks.clear()
                return
            self._done(item_id)
