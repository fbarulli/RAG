"""
DB engine and session management.

Usage:
    from rag_pipeline.db.engine import get_engine, init_db, get_session

    init_db()  # creates tables if not exist
    with get_session() as session:
        session.add(...)
        session.commit()
"""
from contextlib import contextmanager
from sqlmodel import SQLModel, Session, create_engine
from rag_pipeline.core.paths import Paths


def get_engine():
    db_path = Paths.results_db()
    return create_engine(f"sqlite:///{db_path}", echo=False)


def init_db() -> None:
    """Create all tables if they don't exist."""
    from rag_pipeline.db import models  # noqa: F401 — registers SQLModel metadata
    SQLModel.metadata.create_all(get_engine())


@contextmanager
def get_session():
    engine = get_engine()
    with Session(engine) as session:
        yield session
