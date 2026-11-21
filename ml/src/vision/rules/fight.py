"""Драка: два уровня тревоги (ТЗ 6.1) — suspicion, затем confirmation с тем же episode_id.

Каждый кадр:
1. Кладём рамки и скелеты в окно window_s (~2 с).
2. Ищем группы из 2+ близких людей.
3. FightScorer даёт группе score 0..1 за окно (сейчас эвристика, потом модели).
4. Эпизод = группа, у которой score дошёл до suspect_thr:
   - первый раз            -> одно событие suspicion;
   - score >= confirm_thr подряд confirm_s секунд -> одно событие confirmation (тот же episode_id);
   - score < suspect_thr дольше end_s или группа распалась -> эпизод закрыт.
   Без подтверждения ML больше ничего не шлёт: 10-секундный таймер — задача BE.

Эпизод привязан к людям: группа на следующем кадре продолжает эпизод, если в ней
есть хотя бы один его участник.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count

from vision.analytics.fight_scorer import FrameObs, find_groups, make_scorer
from vision.rules.base import Detection, Participant, Rule, RuleContext, register

SUSPICION, CONFIRMATION = "suspicion", "confirmation"


@dataclass
class _Episode:
    episode_id: str
    members: set[int]
    started: datetime
    last_active: datetime                 # последний момент со score >= suspect_thr
    confirm_since: datetime | None = None
    confirmed: bool = False
    last_score: float = 0.0
    seen: set[int] = field(default_factory=set)


@register("fight")
class FightRule(Rule):
    incident_type = "fight"
    requires = {"persons", "poses"}

    def __init__(self, camera_id: str, params: dict):
        super().__init__(camera_id, params)
        p = params
        self.window_s = float(p.get("window_s", 2.0))
        self.suspect_thr = float(p.get("suspect_thr", 0.5))
        self.confirm_thr = float(p.get("confirm_thr", 0.7))
        self.confirm_s = float(p.get("confirm_s", 1.0))
        self.end_s = float(p.get("end_s", 3.0))
        self.close_k = float(p.get("close_k", 1.0))
        self.scorer = make_scorer(p.get("scorer", "heuristic"), p.get("scorer_params"))
        self._window: deque[FrameObs] = deque()
        self._episodes: list[_Episode] = []
        self._seq = count(1)

    def reset(self) -> None:
        self._window.clear()
        self._episodes.clear()

    def phase_of(self, track_id: int) -> str | None:
        """Для отрисовки: suspicion / confirmation, если человек в активном эпизоде."""
        for e in self._episodes:
            if track_id in e.members:
                return CONFIRMATION if e.confirmed else SUSPICION
        return None

    def update(self, ctx: RuleContext) -> list[Detection]:
        now = ctx.ts
        boxes = {t.track_id: t.bbox for t in ctx.tracks}
        confs = {t.track_id: t.conf for t in ctx.tracks}
        self._window.append(FrameObs(now, boxes, dict(ctx.poses or {}),
                                     ctx.image if self.scorer.needs_images else None))
        while self._window and (now - self._window[0].ts).total_seconds() > self.window_s:
            self._window.popleft()

        out: list[Detection] = []
        window = list(self._window)
        for group in find_groups(boxes, self.close_k):
            score = self.scorer.score(window, group, self.window_s)
            ep = self._episode_for(group)
            if ep is None:
                if score < self.suspect_thr:
                    continue
                ep = _Episode(self._new_id(now), set(group), now, now)
                self._episodes.append(ep)
                out.append(self._event(SUSPICION, ep, group, score, ctx, confs))
            ep.members |= set(group)
            ep.last_score = score
            if score >= self.suspect_thr:
                ep.last_active = now
            if ep.confirmed:
                continue
            if score >= self.confirm_thr:
                ep.confirm_since = ep.confirm_since or now
                if (now - ep.confirm_since).total_seconds() >= self.confirm_s:
                    ep.confirmed = True
                    out.append(self._event(CONFIRMATION, ep, group, score, ctx, confs))
            else:
                ep.confirm_since = None

        # закрываем эпизоды, где давно спокойно или людей уже нет
        self._episodes = [e for e in self._episodes
                          if (now - e.last_active).total_seconds() <= self.end_s
                          and e.members & boxes.keys()]
        return out

    def _episode_for(self, group: list[int]) -> _Episode | None:
        for e in self._episodes:
            if e.members & set(group):
                return e
        return None

    def _new_id(self, now: datetime) -> str:
        return f"{self.camera_id}:{now:%Y%m%dT%H%M%S}:{next(self._seq)}"

    def _event(self, kind: str, ep: _Episode, group, score, ctx: RuleContext, confs) -> Detection:
        tracks = {t.track_id: t for t in ctx.tracks}
        return Detection(
            incident_type=self.incident_type, event_type=kind, camera_id=self.camera_id,
            occurred_at=ctx.ts, confidence=round(float(score), 3),
            participants=[Participant(tid, ctx.norm_bbox(tracks[tid].bbox), round(confs[tid], 3))
                          for tid in group if tid in tracks],
            episode_id=ep.episode_id,
        )
