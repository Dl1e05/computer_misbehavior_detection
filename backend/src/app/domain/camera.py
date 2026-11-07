from enum import StrEnum


class CameraSourceType(StrEnum):
    """Where a camera's video comes from (SRS §12.2, ТЗ §12.1)."""

    RTSP = "rtsp"
    FILE = "file"


class CameraStatus(StrEnum):
    """Connection status reported by the media server (SRS §12.2, §13.3)."""

    ONLINE = "online"
    OFFLINE = "offline"
    ERROR = "error"
    DISABLED = "disabled"
