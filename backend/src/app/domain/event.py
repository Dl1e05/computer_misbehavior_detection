from enum import StrEnum


class EventType(StrEnum):
    """Stage of an ML event (SRS §7.2; mapping to transitions is OQ-19)."""

    SUSPICION = "suspicion"
    CONFIRMATION = "confirmation"
    DETECTION = "detection"
