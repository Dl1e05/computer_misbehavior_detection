"""Reader: читает кадры из видеофайла или RTSP-камеры.

Отдаёт Frame(camera_id, ts, image, ...) с частотой process_fps.
- Файл: кадры по порядку, время = старт + номер_кадра / FPS файла.
  realtime=True — в реальном темпе (для демо), False — без ожидания (бенчмарк).
- RTSP: отдельный поток читает камеру и держит только последний кадр,
  поэтому задержка не копится, даже если модель не успевает.
  Время = UTC-момент получения кадра. При обрыве — offline и
  переподключение с паузой 1, 2, 4 … 30 с.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

ONLINE = "online"
OFFLINE = "offline"


@dataclass
class Frame:
    camera_id: str
    ts: datetime          # UTC, с зоной
    image: np.ndarray     # BGR-кадр
    index: int            # номер отданного кадра в текущей сессии
    session: int          # растёт после каждого переподключения -> сбросить трекер и правила
    t_read: float = field(default_factory=time.monotonic)   # когда кадр стал доступен (для задержки, шаг 9)

    @property
    def ts_iso(self) -> str:
        """ISO 8601 с миллисекундами и зоной: 2026-11-21T10:15:30.123+00:00"""
        return self.ts.isoformat(timespec="milliseconds")


def is_live(source: str) -> bool:
    s = source.lower()
    return s.startswith(("rtsp://", "rtsps://", "http://", "https://")) or s.isdigit()


def mask_password(source: str) -> str:
    """rtsp://user:pass@host -> rtsp://user:***@host (для логов)."""
    return re.sub(r"(://[^:/@]+:)[^@]+@", r"\1***@", source)


class VideoSource:
    def __init__(
        self,
        camera_id: str,
        source: str,
        process_fps: float = 8.0,
        realtime: bool = True,
        loop: bool = False,
        fail_threshold: int = 25,
        min_backoff_s: float = 1.0,
        max_backoff_s: float = 30.0,
        timeout_ms: int = 5000,
        on_status: Callable[[str, str], None] | None = None,
        crop: tuple[float, float, float, float] | None = None,
        hold_last_s: float = 0.0,
    ):
        """crop — вырезать часть кадра (x1, y1, x2, y2 в долях 0–1), напр. RGB-половина UR Fall;
        hold_last_s — только файл: после конца ещё столько секунд отдавать последний кадр
        (протокол оценки на коротких клипах: «человек продолжает лежать»)."""
        self.camera_id = camera_id
        self.source = os.path.expandvars(source)   # ${CAM01_PASS} -> из переменных окружения
        self.process_fps = process_fps
        self.realtime = realtime
        self.loop = loop
        self.fail_threshold = fail_threshold
        self.min_backoff_s = min_backoff_s
        self.max_backoff_s = max_backoff_s
        self.timeout_ms = timeout_ms   # открыть / прочитать кадр, иначе read() висит ~30 с
        self.on_status = on_status
        self.live = is_live(self.source)
        self.crop = tuple(crop) if crop else None
        if self.crop and not (0 <= self.crop[0] < self.crop[2] <= 1 and 0 <= self.crop[1] < self.crop[3] <= 1):
            raise ValueError(f"{camera_id}: crop — доли 0–1, x1<x2, y1<y2: {crop}")
        self.hold_last_s = float(hold_last_s)

        self.status = OFFLINE
        self.session = 0
        self._stop = threading.Event()

        # только для live
        self._lock = threading.Lock()
        self._new_frame = threading.Condition(self._lock)
        self._latest: tuple[datetime, np.ndarray, float] | None = None
        self._latest_seq = 0
        self._thread: threading.Thread | None = None

    # ---------- общее ----------

    def _set_status(self, status: str) -> None:
        if status == self.status:
            return
        self.status = status
        log.info("camera %s: %s", self.camera_id, status)
        if self.on_status:
            self.on_status(self.camera_id, status)

    def _open(self) -> cv2.VideoCapture | None:
        src = int(self.source) if self.source.isdigit() else self.source
        if isinstance(src, int):
            cap = cv2.VideoCapture(src)
        elif self.live:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.timeout_ms,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, self.timeout_ms,
            ])
        else:
            cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None
        if self.live:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def stop(self) -> None:
        self._stop.set()
        with self._lock:
            self._new_frame.notify_all()
        if self._thread:
            self._thread.join(timeout=5)

    def __iter__(self) -> Iterator[Frame]:
        frames = self._iter_live() if self.live else self._iter_file()
        return self._cropped(frames) if self.crop else frames

    def _cropped(self, frames: Iterator[Frame]) -> Iterator[Frame]:
        for f in frames:
            h, w = f.image.shape[:2]
            x1, y1, x2, y2 = self.crop
            f.image = np.ascontiguousarray(f.image[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)])
            yield f

    # ---------- файл ----------

    def _iter_file(self) -> Iterator[Frame]:
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                log.error("camera %s: не удалось открыть %s", self.camera_id, self.source)
                self._set_status(OFFLINE)
                return
            self.session += 1
            self._set_status(ONLINE)
            file_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            start = datetime.now(timezone.utc)
            wall_start = time.monotonic()
            step = 1.0 / self.process_fps
            next_t = 0.0          # время в видео, когда отдать следующий кадр
            n_read = 0
            index = 0
            last_img = None
            try:
                while not self._stop.is_set():
                    video_t = n_read / file_fps
                    n_read += 1
                    if video_t + 1e-9 < next_t:
                        # лишний кадр: grab() без перевода в BGR — в ~6 раз дешевле read() (шаг 9)
                        if not cap.grab():
                            break
                        continue
                    ok, img = cap.read()
                    if not ok:
                        break
                    last_img = img
                    next_t += step
                    if self.realtime:
                        delay = video_t - (time.monotonic() - wall_start)
                        if delay > 0:
                            time.sleep(delay)
                    yield Frame(self.camera_id, start + timedelta(seconds=video_t),
                                img, index, self.session)
                    index += 1
                # удержание последнего кадра (только для оценки на коротких клипах)
                end_t = next_t + self.hold_last_s
                while last_img is not None and next_t < end_t - 1e-9 and not self._stop.is_set():
                    if self.realtime:
                        delay = next_t - (time.monotonic() - wall_start)
                        if delay > 0:
                            time.sleep(delay)
                    yield Frame(self.camera_id, start + timedelta(seconds=next_t), last_img.copy(),
                                index, self.session)
                    index += 1
                    next_t += step
            finally:
                cap.release()
            if not self.loop:
                break
        self._set_status(OFFLINE)

    # ---------- RTSP / камера ----------

    def _reader_loop(self) -> None:
        backoff = self.min_backoff_s
        while not self._stop.is_set():
            cap = self._open()
            if cap is None:
                self._set_status(OFFLINE)
                log.warning("camera %s: нет связи с %s, повтор через %.0f с",
                            self.camera_id, mask_password(self.source), backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)
                continue

            fails = 0
            first = True
            while not self._stop.is_set():
                ok, img = cap.read()
                if not ok:
                    fails += 1
                    if fails >= self.fail_threshold:
                        break     # обрыв -> переподключение
                    time.sleep(0.01)
                    continue
                fails = 0
                if first:
                    first = False
                    backoff = self.min_backoff_s
                    with self._lock:
                        self.session += 1
                        self._latest = None
                    self._set_status(ONLINE)
                with self._lock:
                    self._latest = (datetime.now(timezone.utc), img, time.monotonic())
                    self._latest_seq += 1
                    self._new_frame.notify_all()
            cap.release()
            if not self._stop.is_set():
                self._set_status(OFFLINE)
                with self._lock:
                    self._latest = None
                log.warning("camera %s: обрыв, переподключение через %.0f с",
                            self.camera_id, backoff)
                self._stop.wait(backoff)
                backoff = min(backoff * 2, self.max_backoff_s)

    def _iter_live(self) -> Iterator[Frame]:
        self._thread = threading.Thread(target=self._reader_loop,
                                        name=f"reader-{self.camera_id}", daemon=True)
        self._thread.start()
        step = 1.0 / self.process_fps
        next_due = time.monotonic()
        last_seq = 0
        index = 0
        session = 0
        try:
            while not self._stop.is_set():
                # ждём, пока не пора отдавать следующий кадр (process_fps)
                wait = next_due - time.monotonic()
                if wait > 0:
                    self._stop.wait(wait)
                with self._lock:
                    while not self._stop.is_set() and (
                        self._latest is None or self._latest_seq == last_seq
                    ):
                        self._new_frame.wait(timeout=0.5)
                    if self._stop.is_set():
                        break
                    ts, img, t_read = self._latest
                    last_seq = self._latest_seq
                    cur_session = self.session
                if cur_session != session:
                    session, index = cur_session, 0
                next_due = max(next_due + step, time.monotonic())
                yield Frame(self.camera_id, ts, img, index, session, t_read)
                index += 1
        finally:
            self.stop()