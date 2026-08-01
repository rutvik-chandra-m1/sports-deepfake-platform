"""
Declarative base for all ORM models. Every model in app/models/ should
import Base from here — never create a second declarative base, or
Alembic/`Base.metadata.create_all` will not see all tables.
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass
