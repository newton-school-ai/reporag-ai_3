"""Database session factory module.

Provides connection engines and async session management for SQLite and Postgres.
"""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.reporag.config import settings


def get_async_db_url(url: str) -> str:
    """Converts standard database URL to its asynchronous counterpart."""
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "sqlite+aiosqlite:///")
    elif url.startswith("sqlite://"):
        return url.replace("sqlite://", "sqlite+aiosqlite://")
    elif url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://")
    elif url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+asyncpg://")
    return url


# Get the async database URL
ASYNC_DATABASE_URL = get_async_db_url(settings.database_url)

# Connection options
connect_args = {}
if ASYNC_DATABASE_URL.startswith("sqlite"):
    # SQLite async driver (aiosqlite) requires disabling same-thread checks
    connect_args["check_same_thread"] = False

# Create the async database engine
engine = create_async_engine(
    ASYNC_DATABASE_URL,
    connect_args=connect_args,
    echo=False,
)

# Async session factory
async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency for database sessions.

    Yields:
        AsyncSession: An active async database session.
    """
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()
