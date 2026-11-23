"""Detection (от правила) -> Event (контракт); ParticipantMessage (лица) -> Participant.

event_id ставится здесь, один раз: тот же id уходит в Event, в сообщение участника и
при повторе из outbox. run_id — id запуска процесса: после перезапуска track_id
начинаются заново, и BE по run_id понимает, что track 5 «до» и «после» — разные люди.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

from vision.events.schema import Event, EventParticipant, Participant


def new_run_id(now: datetime | None = None) -> str:
    return f"ml-{(now or datetime.now(timezone.utc)):%Y%m%d-%H%M%S}"


class ModelVersions:
    """model_version события — только модели, которые участвовали в этом событии (до 128 символов).

    app.yaml -> versions: {ml, detector, pose, tracker, fire, rules: {incident: версия правила}}
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.rules = cfg.get("rules", {})

    def for_incident(self, incident_type: str, requires: set[str]) -> str:
        c = self.cfg
        parts = [f"ml-{c.get('ml', '0.0.0')}"]
        if "persons" in requires:
            parts += [c.get("detector", "det"), c.get("tracker", "tracker")]
        if "poses" in requires:
            parts.append(c.get("pose", "pose"))
        if "fire" in requires:
            parts.append(c.get("fire", "fire"))
        if incident_type in self.rules:
            parts.append(self.rules[incident_type])
        s = "/".join(parts)
        return s[:128]

    def node(self) -> str:
        c = self.cfg
        return "/".join([f"ml-{c.get('ml', '0.0.0')}", c.get("detector", "det"), c.get("pose", "pose"),
                         c.get("tracker", "tracker")])[:128]


def event_from_detection(d, run_id: str, model_version: str, event_id: uuid.UUID | None = None) -> Event:
    return Event(
        event_id=event_id or uuid.uuid4(),
        event_type=d.event_type,
        incident_type=d.incident_type,
        camera_id=d.camera_id,
        occurred_at=d.occurred_at,
        confidence=min(max(float(d.confidence), 0.0), 1.0),
        participants=[EventParticipant(track_id=int(p.track_id), bbox=tuple(round(v, 4) for v in p.bbox),
                                       confidence=min(max(float(p.confidence), 0.0), 1.0))
                      for p in d.participants],
        model_version=model_version,
        run_id=run_id,
        zone_id=d.zone_id,
        zone_version=d.zone_version,
        episode_id=d.episode_id,
        bbox=tuple(round(v, 4) for v in d.bbox) if d.bbox else None,
    )


def participant_from(m) -> Participant:
    """face.participants.ParticipantMessage -> контрактное сообщение."""
    return Participant(event_id=uuid.UUID(m.event_id), camera_id=m.camera_id, track_id=int(m.track_id),
                       snapshot_ref=m.snapshot_ref, snapshot_quality=m.snapshot_quality,
                       embedding=m.embedding, embedding_model_version=m.embedding_model_version)
