import enum
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Enum, MetaData, func, text
from sqlalchemy.orm import DeclarativeBase, Mapped, declared_attr, mapped_column

# column_0_N_* uses every column of the constraint, so two multi-column
# indexes that start with the same column still get different names.
NAMING_CONVENTION = {
    "ix": "ix_%(column_0_N_label)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "pk": "pk_%(table_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s_%(referred_table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {
        enum.Enum: Enum(
            enum.Enum,
            native_enum=False,
            create_constraint=True,
            values_callable=lambda e: [m.value for m in e],
        ),
        # Every Mapped[datetime] is timestamptz (SRS §12.1), never a naive timestamp.
        datetime: DateTime(timezone=True),
    }

    @declared_attr.directive
    def __mapper_args__(cls: type["Base"]) -> dict[str, Any]:
        # eager_defaults: fetch server-generated values (updated_at, version)
        # with RETURNING, so reading them after a flush never triggers a lazy
        # load, which would fail under AsyncSession.
        args: dict[str, Any] = {"eager_defaults": True}
        if issubclass(cls, VersionMixin):
            # Optimistic locking (SRS §12.1): every UPDATE increments version
            # and only succeeds if the row still has the version that was read.
            args["version_id_col"] = cls.__table__.c.version
        return args


class IdMixin:
    id: Mapped[uuid.UUID] = mapped_column(
        primary_key=True, server_default=text("gen_random_uuid()")
    )


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now()
    )


class VersionMixin:
    version: Mapped[int] = mapped_column(server_default=text("1"))


class EntityMixin(IdMixin, TimestampMixin, VersionMixin):
    """Standard columns for a mutable table with a UUID key (SRS §12.1)."""
