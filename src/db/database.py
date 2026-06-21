"""
db/database.py
---------------
Engine + session factory. SQLite file by default (zero setup for the demo);
swapping DB_URL to a Postgres DSN in config.py is the only change needed
to go to production, since all access goes through SQLAlchemy.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from contextlib import contextmanager

from src.config import get_config
from src.db.models import Base

_config = get_config()

engine = create_engine(
    _config.DB_URL,
    connect_args={"check_same_thread": False} if _config.DB_URL.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


@contextmanager
def get_session():
    """Context-managed session: guarantees commit/rollback/close happen
    even if the caller raises, which is the #1 source of connection leaks
    in hand-rolled DB code."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
