"""Telling a database outage apart from a bad message (used by both workers)."""

from sqlalchemy.exc import DBAPIError, OperationalError

# SQLSTATE classes that say nothing about the message: connection exception,
# insufficient resources (e.g. too many connections), operator intervention
# (e.g. the database is starting up or shutting down).
_UNAVAILABLE_SQLSTATE_CLASSES = frozenset({"08", "53", "57"})


def is_database_unavailable(error: Exception) -> bool:
    """True if the database failed, not the message: retry without limit."""
    # A refused connection arrives as a plain OSError, not wrapped by
    # SQLAlchemy.
    if isinstance(error, OSError):
        return True
    if not isinstance(error, DBAPIError):
        return False
    # A connection lost during a query; asyncpg reports it as InterfaceError.
    # An InterfaceError without this is a bug and is counted.
    if error.connection_invalidated or isinstance(error, OperationalError):
        return True
    sqlstate = getattr(error.orig, "sqlstate", None)
    return isinstance(sqlstate, str) and sqlstate[:2] in _UNAVAILABLE_SQLSTATE_CLASSES


def describe(error: Exception) -> str:
    """Class names only: an SQL error's message holds its parameters."""
    if isinstance(error, DBAPIError):
        sqlstate = getattr(error.orig, "sqlstate", None)
        return f"{type(error).__name__}: {type(error.orig).__name__} {sqlstate or ''}"
    return type(error).__name__
