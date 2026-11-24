"""Конвейер узла v1: Reader -> Inference (батч) -> Analytics (трекер + правила + лица) -> Publisher.

Один и тот же код для демо (scripts/run_tracking.py) и замеров (scripts/load_test.py).

    node = Pipeline(app_cfg, cams, zones, root)      # модели, трекеры, правила, лица, MQTT
    bus = FrameBus(node.sources).start()
    while not bus.done:
        for out in node.process(bus.next_batch()):    # out: tracks, poses, phases, events кадра
            ...
    node.close()

Замеры по стадиям копятся в node.metrics (мс на батч, задержка кадра от чтения до конца правил).
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from vision.analytics.tracker import CameraTracker
from vision.events.builder import ModelVersions, event_from_detection, new_run_id, participant_from
from vision.events.schema import is_agreed
from vision.events.outbox import Outbox
from vision.events.publisher import Publisher
from vision.face.participants import FACE_EVENTS, ParticipantCollector
from vision.inference.detector import BatchDetector
from vision.inference.fire import FireDetector
from vision.inference.pose import PoseEstimator, attach_poses
from vision.rules import RuleContext, build_rules, required_inputs
from vision.runtime.status import NodeStatus
from vision.sources.video_source import Frame, VideoSource

log = logging.getLogger("pipeline")


@dataclass
class FrameOut:
    frame: Frame
    tracks: list
    poses: dict | None
    events: list                                   # контрактные Event этого кадра
    rules: list                                    # правила камеры (для отрисовки зон/фаз)


@dataclass
class Metrics:
    """Сырые замеры; агрегирует load_test.py."""
    batch_size: list[int] = field(default_factory=list)
    det_ms: list[float] = field(default_factory=list)
    pose_ms: list[float] = field(default_factory=list)
    fire_ms: list[float] = field(default_factory=list)
    analytics_ms: list[float] = field(default_factory=list)     # трекеры + правила + лица, весь батч
    latency_ms: list[tuple[float, float]] = field(default_factory=list)   # (момент, задержка кадра)
    frames: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    events: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    participants: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    local_only: dict[str, int] = field(default_factory=lambda: defaultdict(int))   # не отправлено: тип не согласован

    def reset(self) -> None:
        self.__init__()


class Pipeline:
    def __init__(self, app: dict, cams: list[dict], zones: dict, root: Path, *, fast: bool = False,
                 fps: float | None = None, faces: bool = True, mqtt: bool = True,
                 on_event=None, on_participant=None, shared: "Pipeline | None" = None):
        """app — app.yaml; cams — камеры (как в cameras.yaml); fast — файлы без ожидания и без loop.

        shared — взять уже загруженные модели у другого Pipeline (оценка пачками роликов).
        """
        self.app, self.cams, self.root = app, cams, root
        self.on_event, self.on_participant = on_event, on_participant
        self.status = NodeStatus(node_id=app.get("node", {}).get("node_id", "ml-node-1"))
        self.metrics = Metrics()
        self.sources, self.trackers, self.rules = [], {}, {}

        for cam in cams:
            cid = cam["id"]
            cam_fps = fps or cam.get("process_fps", 8)
            src = str(cam["source"])
            if not src.startswith(("rtsp", "http")) and not Path(src).is_absolute():
                src = str(root / src)
            self.sources.append(VideoSource(cid, src, process_fps=cam_fps,
                                            realtime=not fast and cam.get("realtime", True),
                                            loop=cam.get("loop", False) and not fast,
                                            on_status=self.status.set_camera,
                                            crop=cam.get("crop"), hold_last_s=cam.get("hold_last_s", 0.0)))
            self.status.set_camera(cid, "offline")
            self.trackers[cid] = CameraTracker.from_config(cid, cam_fps, app["tracker"])
            self.rules[cid] = build_rules(cid, cam.get("rules"), zones)
            for r in self.rules[cid]:
                self.status.set_rule(cid, r.incident_type, "on")
            log.info("camera %s: правила %s", cid, [type(r).__name__ for r in self.rules[cid]])

        self._init_models(shared)
        self._init_faces(faces)
        self._init_publisher(mqtt)
        self._sessions: dict[str, int] = {}

    # ---------- сборка ----------

    def _weights(self, cfg: dict) -> dict:
        cfg = dict(cfg)
        cfg["weights"] = str(self.root / cfg["weights"])
        return cfg

    def _init_models(self, shared: "Pipeline | None" = None) -> None:
        n = len(self.cams)
        if shared is not None:
            self.detector = shared.detector
        else:
            self.detector = BatchDetector.from_config(self._weights(self.app["detector"]))
            self.detector.warmup(n)

        # огонь: только камеры с правилом fire; нет весов — правило выключено, узел работает
        self.fire_cams = {c for c, rs in self.rules.items() if "fire" in required_inputs(rs)}
        self.fire_model, self.fire_period = None, 1.0
        self._last_fire: dict[str, object] = {}
        if self.fire_cams:
            cfg = self._weights(self.app["fire"])
            self.fire_period = 1.0 / float(cfg.pop("run_hz", 3))
            if shared is not None and shared.fire_model is not None:
                self.fire_model = shared.fire_model
            else:
                self.fire_model = FireDetector.from_config(cfg)
                if self.fire_model.available:
                    self.fire_model.warmup(len(self.fire_cams))
            if not self.fire_model.available:
                for c in self.fire_cams:
                    self.rules[c] = [r for r in self.rules[c] if "fire" not in r.requires]
                    self.status.set_rule(c, "fire", f"off: {self.fire_model.reason}")
                log.warning("правило fire выключено на %s: %s", sorted(self.fire_cams), self.fire_model.reason)
                self.fire_cams = set()

        # поза: только камеры, где правилам нужны скелеты
        self.pose_cams = {c for c, rs in self.rules.items() if "poses" in required_inputs(rs)}
        self.pose_model, self.match_iou = None, 0.5
        if self.pose_cams:
            cfg = self._weights(self.app["pose"])
            cfg.pop("mode", None)
            self.match_iou = cfg.pop("match_iou", 0.5)
            if shared is not None and shared.pose_model is not None:
                self.pose_model = shared.pose_model
            else:
                self.pose_model = PoseEstimator.from_config(cfg)
                self.pose_model.warmup(len(self.pose_cams))

    def _init_faces(self, enabled: bool) -> None:
        from vision.face.face_service import FaceService
        from vision.face.storage import LocalSnapshotStorage
        from vision.face.track_buffer import TrackBuffer

        incidents = {inc for inc, _ in FACE_EVENTS}
        face_cams = {c for c, rs in self.rules.items() if any(r.incident_type in incidents for r in rs)}
        self.collectors: dict[str, ParticipantCollector] = {}
        if not face_cams:
            return
        if not enabled:
            for c in face_cams:
                self.status.set_rule(c, "faces", "off: выключено при запуске")
            return
        cfg = dict(self.app["face"])
        service = FaceService.from_config(cfg)
        storage = LocalSnapshotStorage(self.root / cfg.get("storage", "data/snapshots"))
        for c in face_cams:
            buf = TrackBuffer(keep_s=float(cfg.get("buffer_s", 5)), every_n=int(cfg.get("every_n", 3)),
                              upper_frac=float(cfg.get("upper_frac", 0.5)),
                              lost_s=float(self.app["tracker"].get("lost_s", 2.0)))
            self.collectors[c] = ParticipantCollector(c, service, storage, buf, float(cfg.get("after_s", 2)))
            self.status.set_rule(c, "faces", f"on ({service.model_version})")

    def _init_publisher(self, enabled: bool) -> None:
        self.run_id = new_run_id(self.status.node_id)
        self.versions = ModelVersions(self.app.get("versions", {}))
        cfg = dict(self.app.get("mqtt", {}))
        if cfg.get("dropped_log"):
            cfg["dropped_log"] = str(self.root / cfg["dropped_log"])
        self.send_proposed = bool(cfg.get("send_proposed", False))
        self.publisher = None
        if enabled and cfg.get("enabled", True):
            outbox = Outbox(self.root / cfg.get("outbox", "data/outbox.sqlite"))
            self.publisher = Publisher.from_config({**cfg, **self.app.get("node", {})}, self.run_id, outbox,
                                                   status_fn=self.status.snapshot,
                                                   model_version=self.versions.node()).start()
            log.info("MQTT: %s:%s, node %s, run %s, в outbox с прошлых запусков: %d",
                     self.publisher.host, self.publisher.port, self.publisher.node_id, self.run_id, len(outbox))
        else:
            log.info("MQTT выключен — события только локально (run %s)", self.run_id)

    # ---------- работа ----------

    def process(self, batch: list[Frame]) -> list[FrameOut]:
        if not batch:
            return []
        m = self.metrics
        m.batch_size.append(len(batch))
        t0 = time.perf_counter()
        dets = self.detector([f.image for f in batch])                 # ОДИН прогон на все камеры
        m.det_ms.append((time.perf_counter() - t0) * 1000)

        need = [i for i, f in enumerate(batch) if f.camera_id in self.pose_cams]
        pose_by_idx = {}
        if need:                                                       # поза — вторым батчем, где нужна
            t0 = time.perf_counter()
            for i, ps in zip(need, self.pose_model([batch[i].image for i in need])):
                pose_by_idx[i] = ps
            m.pose_ms.append((time.perf_counter() - t0) * 1000)

        due = [i for i, f in enumerate(batch) if f.camera_id in self.fire_cams and (
            f.camera_id not in self._last_fire
            or (f.ts - self._last_fire[f.camera_id]).total_seconds() >= self.fire_period)]
        fire_by_idx = {}
        if due:                                                        # огонь — третьим, run_hz раз в секунду
            t0 = time.perf_counter()
            for i, boxes in zip(due, self.fire_model([batch[i].image for i in due])):
                fire_by_idx[i] = boxes
                self._last_fire[batch[i].camera_id] = batch[i].ts
            m.fire_ms.append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        outs = [self._analytics(f, d, pose_by_idx.get(idx), fire_by_idx.get(idx))
                for idx, (f, d) in enumerate(zip(batch, dets))]
        m.analytics_ms.append((time.perf_counter() - t0) * 1000)
        now = time.monotonic()
        for f in batch:
            m.latency_ms.append((now, (now - f.t_read) * 1000))
        return outs

    def _analytics(self, f: Frame, dets, poses_raw, fire) -> FrameOut:
        cid = f.camera_id
        tracks = self.trackers[cid].update(dets, f.image, session=f.session)
        self.metrics.frames[cid] += 1
        cam_rules = self.rules[cid]
        col = self.collectors.get(cid)
        if self._sessions.get(cid) not in (None, f.session):
            for r in cam_rules:
                r.reset()                                              # после переподключения id недействительны
            if col:
                self._emit_participants(cid, col.reset())
        self._sessions[cid] = f.session
        if col:                                                        # кропы в память; лица — только по событию
            self._emit_participants(cid, col.on_frame(f.ts, f.image, tracks))

        h, w = f.image.shape[:2]
        poses = attach_poses(tracks, poses_raw, self.match_iou) if poses_raw is not None else None
        ctx = RuleContext(cid, f.ts, w, h, tracks, session=f.session, poses=poses, image=f.image, fire=fire,
                          fire_names=self.fire_model.names if self.fire_model else None)
        events = []
        for r in cam_rules:
            for d in r.update(ctx):
                self.metrics.events[cid] += 1
                e = self._emit_event(r, d)
                if e is not None:
                    events.append(e)
                    if col:
                        col.on_event(d, str(e.event_id))               # сам проверит, подтверждающее ли
        return FrameOut(f, tracks, poses, events, cam_rules)

    def _emit_event(self, rule, d):
        try:
            e = event_from_detection(d, self.run_id, self.versions.for_incident(d.incident_type, rule.requires))
        except ValueError as err:
            log.error("событие %s/%s не по контракту, не отправлено: %s", d.incident_type, d.event_type, err)
            return None
        if self.publisher:
            if is_agreed(e.incident_type) or self.send_proposed:
                self.publisher.submit_event(e)
            else:                                                      # тип ещё не согласован с BE
                self.metrics.local_only[e.incident_type] += 1
        if self.on_event:
            self.on_event(e)
        return e

    def _emit_participants(self, cid: str, msgs) -> None:
        for msg in msgs:
            self.metrics.participants[cid]["face" if msg.embedding else "no_usable_face"] += 1
            try:
                p = participant_from(msg)
            except ValueError as err:
                log.error("участник не по контракту: %s", err)
                continue
            if self.publisher:
                self.publisher.submit_participant(p)
            if self.on_participant:
                self.on_participant(p, msg.reason)

    def close(self) -> None:
        for cid, col in self.collectors.items():
            self._emit_participants(cid, col.flush())
        if self.publisher:
            self.publisher.stop(flush_timeout=5)
