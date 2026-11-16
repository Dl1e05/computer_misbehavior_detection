from enum import StrEnum


class AuditResult(StrEnum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    DENIED = "DENIED"


class AuditObjectType(StrEnum):
    INCIDENT = "incident"
    PARTICIPANT = "participant"
    IDENTIFICATION = "identification"
    PERSON = "person"
    CONSENT = "consent"
    CLIP = "clip"
    CAMERA = "camera"
    ZONE = "zone"
    EVENT = "event"
