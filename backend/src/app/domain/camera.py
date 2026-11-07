from enum import StrEnum


class CameraSourceType(StrEnum):
    """Where a camera's video comes from (SRS §12.2, ТЗ §12.1)."""

    RTSP = "rtsp"
    FILE = "file"


class CameraStatus(StrEnum):
    """Connection status reported by the media server (SRS §12.2, §13.3).

    Only the connection state. Whether a camera is switched on by the
    Administrator is Camera.enabled (BE-FR-60), the single source of truth;
    the SRS's proposed "disabled" status value is therefore not used.
    """

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
