import os

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def get_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        return "sqlite+aiosqlite:///reporag.db"

    # Auto-convert postgresql:// to postgresql+asyncpg:// for async compatibility
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)

    return url

engine = create_async_engine(get_database_url(), echo=False)
AsyncSessionLocal = async_sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)

async def get_db():
    async with AsyncSessionLocal() as session:
        yield session
