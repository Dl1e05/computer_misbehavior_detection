from sqlalchemy import text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, EntityMixin
from app.domain.incident import Role


class User(EntityMixin, Base):
    __tablename__ = "users"

    login: Mapped[str] = mapped_column(unique=True)
    password_hash: Mapped[str]
    role: Mapped[Role]
    active: Mapped[bool] = mapped_column(server_default=text("true"))
