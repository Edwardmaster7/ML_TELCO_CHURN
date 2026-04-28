from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import SessionLocal

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependência para obter uma sessão assíncrona do banco de dados."""
    async with SessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
