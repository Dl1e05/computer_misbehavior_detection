"""Саботаж камеры: закрыли объектив, развернули, расфокусировали, засветили.

Работает по самой картинке, без нейросетей и без людей: check_hz раз в секунду кадр уменьшается
до ~160 пикселей в ширину (это дёшево), и сравнивается с эталоном сцены.
- эталон — медиана первых learn_s секунд НОРМАЛЬНЫХ кадров (чёрные/однородные/засвеченные
  пропускаются: камера могла включиться закрытой); потом медленно подстраивается под свет,
  пока всё нормально и в кадре нет людей;
- поворотные (PTZ) камеры и камеры на движущихся объектах — правило не включать: «moved» там норма;
- glare     — больше glare_frac пикселей пересвечено (фонарик в объектив) — проверяется первым:
               белый однородный кадр — это засветка, а не «закрыли»;
- covered   — картинка почти однородная (std яркости < flat_std): рука, бумага, краска;
- defocused — резкость (дисперсия Лапласиана) упала ниже blur_ratio от эталона
               (если эталон сам резкий, иначе не проверяем);
- moved     — контуры сцены почти не совпадают с эталоном (корреляция карт градиентов < moved_corr):
               камеру развернули (замер: сдвиг/разворот 0.02–0.22; толпа на demo1 0.42–0.71, человек 0.88).
Состояние должно держаться hold_s подряд -> одно событие на эпизод; снова — после clear_s нормы.
Если камеру развернули и так оставили, через relearn_s новый вид становится эталоном.
Известное ограничение: выключенный свет похож на «закрыли» (тёмный однородный кадр).
Событие: incident camera_tampering, participants пустой, extra: tamper_type.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import cv2
import numpy as np

from vision.rules.base import Detection, Rule, RuleContext, register

TYPES = ("covered", "glare", "defocused", "moved")


@dataclass
class Measure:
    mean: float
    std: float
    saturated: float
    sharp: float
    grad: np.ndarray


def measure(image: np.ndarray, width: int = 160) -> tuple[Measure, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    h, w = gray.shape[:2]
    small = cv2.resize(gray, (width, max(1, round(width * h / w))), interpolation=cv2.INTER_AREA)
    f = small.astype(np.float32)
    gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
    grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (5, 5), 0)
    return Measure(float(f.mean()), float(f.std()), float((small >= 250).mean()),
                   float(cv2.Laplacian(small, cv2.CV_64F).var()), grad), f


def correlation(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-6 else 0.0


@register("camera_tampering")
class CameraTamperingRule(Rule):
    incident_type = "camera_tampering"
    requires = {"image"}

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        p = params
        self.check_hz = float(p.get("check_hz", 2.0))
        self.learn_s = float(p.get("learn_s", 5.0))
        self.hold_s = float(p.get("hold_s", 3.0))
        self.clear_s = float(p.get("clear_s", 5.0))
        self.relearn_s = float(p.get("relearn_s", 60.0))
        self.flat_std = float(p.get("flat_std", 10.0))
        self.glare_frac = float(p.get("glare_frac", 0.4))
        self.blur_ratio = float(p.get("blur_ratio", 0.35))
        self.min_ref_sharp = float(p.get("min_ref_sharp", 30.0))
        self.moved_corr = float(p.get("moved_corr", 0.3))
        self.adapt = float(p.get("adapt", 0.02))
        self.reset()

    def reset(self) -> None:
        self._last_check: datetime | None = None
        self._learn: list[tuple[np.ndarray, Measure]] = []
        self._learn_start: datetime | None = None
        self.ref_gray: np.ndarray | None = None
        self.ref: Measure | None = None
        self._cand: str | None = None
        self._cand_since: datetime | None = None
        self._alerted: str | None = None
        self._normal_since: datetime | None = None
        self.state = "learning"                     # для отрисовки/отладки

    def classify(self, m: Measure) -> tuple[str | None, float]:
        """-> (тип саботажа или None, сила 0..1)."""
        if m.saturated >= self.glare_frac:
            return "glare", m.saturated
        if m.std < self.flat_std:
            return "covered", 1.0 - m.std / self.flat_std
        if self.ref.sharp >= self.min_ref_sharp and m.sharp < self.blur_ratio * self.ref.sharp:
            return "defocused", 1.0 - m.sharp / self.ref.sharp
        corr = correlation(m.grad, self.ref.grad)
        if corr < self.moved_corr:
            return "moved", 1.0 - max(corr, 0.0)
        return None, 0.0

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        if ctx.image is None or (self._last_check is not None
                                 and (now - self._last_check).total_seconds() < 1.0 / self.check_hz):
            return []
        self._last_check = now
        m, gray = measure(ctx.image)
        if self.ref is None:                                          # учим эталон сцены
            if m.std < self.flat_std or m.saturated >= self.glare_frac:
                return []                                             # чёрный/однородный/засвеченный кадр — не эталон
            self._learn_start = self._learn_start or now
            self._learn.append((gray, m))
            if (now - self._learn_start).total_seconds() >= self.learn_s and len(self._learn) >= 3:
                self._set_ref(np.median(np.stack([g for g, _ in self._learn]), axis=0).astype(np.float32))
                self._learn.clear()
                self.state = "normal"
            return []

        kind, strength = self.classify(m)
        if kind is None:
            self._cand, self._cand_since = None, None
            self._normal_since = self._normal_since or now
            if self._alerted and (now - self._normal_since).total_seconds() >= self.clear_s:
                self._alerted = None
            if not self._alerted and not ctx.tracks:                  # подстройка под медленные изменения света
                self._set_ref((1 - self.adapt) * self.ref_gray + self.adapt * gray)
            self.state = "normal"
            return []

        self._normal_since = None
        if kind != self._cand:
            self._cand, self._cand_since = kind, now
        held = (now - self._cand_since).total_seconds()
        self.state = kind
        if kind == "moved" and self._alerted == "moved" and held >= self.relearn_s:
            self._set_ref(gray)                                       # камеру так и оставили — новый вид = эталон
            self._alerted, self._cand = None, None
            return []
        if held >= self.hold_s and self._alerted != kind:
            self._alerted = kind
            return [Detection(incident_type=self.incident_type, event_type="detection", camera_id=self.camera_id,
                              occurred_at=now, confidence=round(float(np.clip(strength, 0.0, 1.0)), 3),
                              participants=[], extra={"tamper_type": kind})]
        return []

    def _set_ref(self, gray: np.ndarray) -> None:
        self.ref_gray = gray.astype(np.float32)
        small = np.clip(self.ref_gray, 0, 255).astype(np.uint8)
        f = self.ref_gray
        gx, gy = cv2.Sobel(f, cv2.CV_32F, 1, 0), cv2.Sobel(f, cv2.CV_32F, 0, 1)
        self.ref = Measure(float(f.mean()), float(f.std()), float((small >= 250).mean()),
                           float(cv2.Laplacian(small, cv2.CV_64F).var()),
                           cv2.GaussianBlur(cv2.magnitude(gx, gy), (5, 5), 0))
