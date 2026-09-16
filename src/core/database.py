"""
GANESHA V1 — Database Connection & Session Management

Provides synchronous database sessions using SQLAlchemy.
All database interactions should use the get_db_session() context manager.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
from loguru import logger

from src.core.config import settings


# ── Engine Creation ──
engine = create_engine(
    settings.database_url,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # Verify connections before checkout
    echo=False,          # Set True temporarily for SQL debugging
)

# ── Session Factory ──
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """
    Provides a transactional database session.

    Usage:
        with get_db_session() as db:
            db.execute(text("SELECT 1"))
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def verify_database_connection() -> bool:
    """
    Test the database connection at startup.
    Returns True if connection is successful, False otherwise.
    """
    try:
        with get_db_session() as db:
            result = db.execute(text("SELECT 1"))
            row = result.scalar()
            if row == 1:
                logger.success(f"Database connection verified: {settings.db_name}@{settings.db_host}")
                return True
    except Exception as e:
        logger.error(f"Database connection FAILED: {e}")
    return False
