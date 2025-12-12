import sys
import logging
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import declarative_base, sessionmaker

from src.config import settings

logger = logging.getLogger(__name__)

engine = create_async_engine(str(settings.DATABASE_URL), pool_pre_ping=True)

AsyncSessionFactory = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

Base = declarative_base()


async def get_db():
    """FastAPI dependency to get a DB session."""
    async with AsyncSessionFactory() as session:
        yield session


async def test_db_connection():
    """
    Tests the database connection.
    Returns True on success, raises exception on failure.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
            logger.debug("Database connection successful.")
            return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}", exc_info=True)
        raise
