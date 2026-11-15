from enum import StrEnum


class NotificationKind(StrEnum):
    SUSPICION = "suspicion"
    ALARM = "alarm"
    ESCALATION = "escalation"
    INCIDENT_UPDATE = "incident_update"
    CAMERA_STATUS = "camera_status"
    ANALYTICS_STATUS = "analytics_status"


# Kinds about an incident: they carry incident_id. The others (camera and
# analytics status) are system messages without one.
INCIDENT_NOTIFICATION_KINDS: frozenset[NotificationKind] = frozenset(
    {
        NotificationKind.SUSPICION,
        NotificationKind.ALARM,
        NotificationKind.ESCALATION,
        NotificationKind.INCIDENT_UPDATE,
    }
)
