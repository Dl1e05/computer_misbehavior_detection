"""Внутренняя шина v1: Reader-потоки всех камер -> батч для Inference.

Каждая камера читается в своём потоке (VideoSource уже отдаёт кадры с её
process_fps). Шина хранит по камере только последний непрочитанный кадр:
если модель не успевает, старые кадры выбрасываются, задержка не растёт.

drop_old=False — для бенчмарка по файлам: Reader ждёт, пока кадр заберут,
ни один кадр не теряется (у живой камеры так нельзя — копится задержка).

v2: потоки заменяются процессами, интерфейс next_batch() остаётся тем же.
"""
from __future__ import annotations

import threading
import time

from vision.sources.video_source import Frame, VideoSource


class FrameBus:
    def __init__(self, sources: list[VideoSource], drop_old: bool = True):
        self.sources = sources
        self.drop_old = drop_old
        self._latest: dict[str, Frame] = {}
        self._finished: set[str] = set()
        self._cond = threading.Condition()
        self._threads: list[threading.Thread] = []
        self._stop = threading.Event()

    def start(self) -> "FrameBus":
        for src in self.sources:
            t = threading.Thread(target=self._pump, args=(src,),
                                 name=f"bus-{src.camera_id}", daemon=True)
            t.start()
            self._threads.append(t)
        return self

    def _pump(self, src: VideoSource) -> None:
        try:
            for frame in src:
                if self._stop.is_set():
                    break
                with self._cond:
                    while (not self.drop_old and src.camera_id in self._latest
                           and not self._stop.is_set()):
                        self._cond.wait(0.5)              # ждём, пока заберут прошлый кадр
                    self._latest[src.camera_id] = frame   # иначе старый непрочитанный затираем
                    self._cond.notify_all()
        finally:
            with self._cond:
                self._finished.add(src.camera_id)
                self._cond.notify_all()

    @property
    def done(self) -> bool:
        """Все источники закончились (файлы без loop) и кадров не осталось."""
        with self._cond:
            return len(self._finished) == len(self.sources) and not self._latest

    def next_batch(self, timeout: float = 1.0) -> list[Frame]:
        """Ждёт хотя бы один новый кадр и забирает свежие кадры всех камер разом."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while not self._latest and not self._stop.is_set():
                if len(self._finished) == len(self.sources):
                    break
                left = deadline - time.monotonic()
                if left <= 0:
                    break
                self._cond.wait(left)
            batch = list(self._latest.values())
            self._latest.clear()
            self._cond.notify_all()                       # разбудить Reader-ы в режиме drop_old=False
        return batch

    def stop(self) -> None:
        self._stop.set()
        for src in self.sources:
            src.stop()
        with self._cond:
            self._cond.notify_all()
        for t in self._threads:
            t.join(timeout=5)
