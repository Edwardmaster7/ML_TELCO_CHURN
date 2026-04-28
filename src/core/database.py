from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from src.core.config import CONFIG

# Engine assíncrono para o banco de dados
engine = create_async_engine(CONFIG.database_url, echo=False)

# Factory de sessão assíncrona
SessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)
