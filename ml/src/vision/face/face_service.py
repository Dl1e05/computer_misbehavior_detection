"""FaceService: из набора кропов одного человека — лучшее лицо и его embedding.

Сам по себе сервис ничего не решает о приватности: вызывать его можно ТОЛЬКО из
ParticipantCollector (после подтверждённого события) и из scripts/enroll_faces.py.
Тест test_face.py::test_privacy_* следит, что в обычном потоке вызовов 0.

Вектор: 512 чисел ArcFace (w600k_r50 из buffalo_l), normed_embedding — длина 1,
косинусное сходство = скалярное произведение. Версия вектора — model_version;
BE не сравнивает векторы разных версий.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import cv2
import numpy as np

from vision.face.quality import QualityParams, face_quality

NO_USABLE_FACE = "no_usable_face"


@dataclass
class FaceResult:
    quality: float | str                    # 0..1 или "no_usable_face"
    embedding: np.ndarray | None = None     # (512,), float32, норма 1
    jpeg: bytes | None = None               # снимок лица (с полями), для хранилища
    reason: str | None = None               # почему нет лица: no_crops / no_face / too_small / ...
    checked: int = 0                        # сколько кропов просмотрели
    rejected: dict[str, int] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.embedding is not None


def face_jpeg(image: np.ndarray, bbox, margin: float = 0.3, quality: int = 90) -> bytes:
    """Вырезает лицо с полями (margin от размера) и кодирует в JPEG."""
    h, w = image.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in bbox]
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    roi = image[max(0, int(y1 - my)):min(h, int(y2 + my)), max(0, int(x1 - mx)):min(w, int(x2 + mx))]
    ok, buf = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


class FaceService:
    def __init__(self, app, model_version: str = "insightface-buffalo_l-1.0",
                 quality: QualityParams | None = None, min_quality: float = 0.0, max_crops: int = 16):
        """app — объект с методом get(image) -> [face] (InsightFace FaceAnalysis или заглушка)."""
        self.app = app
        self.model_version = model_version
        self.qp = quality or QualityParams()
        self.min_quality = min_quality
        self.max_crops = max_crops
        self.calls = 0                      # сколько раз запускали поиск лиц (для теста приватности)

    @classmethod
    def from_config(cls, cfg: dict) -> "FaceService":
        from insightface.app import FaceAnalysis

        device = cfg.get("device", 0)
        providers = (["CPUExecutionProvider"] if device == "cpu"
                     else ["CUDAExecutionProvider", "CPUExecutionProvider"])
        app = FaceAnalysis(name=cfg.get("model", "buffalo_l"), providers=providers,
                           allowed_modules=["detection", "recognition"])   # без пола/возраста
        det = int(cfg.get("det_size", 320))
        app.prepare(ctx_id=-1 if device == "cpu" else int(device), det_size=(det, det))
        qp = QualityParams(**{k: v for k, v in (cfg.get("quality") or {}).items()})
        return cls(app, cfg.get("model_version", "insightface-buffalo_l-1.0"), qp,
                   float(cfg.get("min_quality", 0.0)), int(cfg.get("max_crops", 16)))

    def best_face(self, images: Iterable[np.ndarray]) -> FaceResult:
        """Ищет лица во всех кропах, берёт одно лучшее по качеству и считает только его вектор."""
        images = [im for im in images if im is not None and im.size]
        if not images:
            return FaceResult(NO_USABLE_FACE, reason="no_crops")
        if len(images) > self.max_crops:                     # равномерно прореживаем
            idx = np.linspace(0, len(images) - 1, self.max_crops).round().astype(int)
            images = [images[i] for i in idx]
        self.calls += 1
        best, best_q, best_img = None, -1.0, None
        rejected: dict[str, int] = {}
        for im in images:
            faces = self.app.get(im)
            if not faces:
                rejected["no_face"] = rejected.get("no_face", 0) + 1
            for f in faces:
                q, why = face_quality(f, im, self.qp)
                if why:
                    rejected[why] = rejected.get(why, 0) + 1
                    continue
                if q > best_q:
                    best, best_q, best_img = f, q, im
        if best is None or best_q < self.min_quality:
            reason = "low_quality" if best is not None else (max(rejected, key=rejected.get) if rejected else "no_face")
            return FaceResult(NO_USABLE_FACE, reason=reason, checked=len(images), rejected=rejected)
        emb = getattr(best, "normed_embedding", None)
        if emb is None:                                      # распознавание не отработало
            return FaceResult(NO_USABLE_FACE, reason="no_embedding", checked=len(images), rejected=rejected)
        emb = np.asarray(emb, dtype=np.float32).reshape(-1)
        return FaceResult(best_q, emb, face_jpeg(best_img, best.bbox), checked=len(images), rejected=rejected)

    def enroll(self, image: np.ndarray, pick: str = "single") -> FaceResult:
        """Лицо с фото (не из потока камеры): для Базы лиц и офлайн-калибровки порога.

        pick="single" — База лиц: на фото должно быть ровно одно пригодное лицо;
        pick="center" — калибровка на датасетах вроде LFW: берём лицо ближе всех к центру
        (на заднем плане бывают другие люди).
        """
        self.calls += 1
        faces = self.app.get(image)
        if not faces:
            return FaceResult(NO_USABLE_FACE, reason="no_face", checked=1)
        scored = [(face_quality(f, image, self.qp), f) for f in faces]
        good = [(q, f) for (q, why), f in scored if why is None]
        if not good:
            return FaceResult(NO_USABLE_FACE, reason=scored[0][0][1], checked=1)
        if len(good) > 1 and pick == "single":
            return FaceResult(NO_USABLE_FACE, reason="several_faces", checked=1)
        h, w = image.shape[:2]

        def center_dist(item):
            x1, y1, x2, y2 = [float(v) for v in item[1].bbox]
            return ((x1 + x2) / 2 - w / 2) ** 2 + ((y1 + y2) / 2 - h / 2) ** 2

        q, f = min(good, key=center_dist)
        if q < self.min_quality:
            return FaceResult(NO_USABLE_FACE, reason="low_quality", checked=1)
        emb = np.asarray(f.normed_embedding, dtype=np.float32).reshape(-1)
        return FaceResult(q, emb, face_jpeg(image, f.bbox), checked=1)
