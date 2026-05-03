from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from src.core.config import CONFIG


class Base(DeclarativeBase):
    pass


class PredictionLog(Base):
    """Registro de cada inferência realizada pela API.

    Campos de feedback (``actual_churn``, ``feedback_at``) são preenchidos
    a posteriori via endpoint ``POST /api/v1/feedback/{customer_id}``.
    """

    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    customer_id = Column(String(50), index=True, nullable=False)
    churn_probability = Column(Float, nullable=False)
    churn_prediction = Column(Integer, nullable=False)  # 0 ou 1
    predicted_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    actual_churn = Column(Integer, nullable=True)       # preenchido via feedback
    feedback_at = Column(DateTime, nullable=True)
    model_version = Column(String(100), nullable=False, default="")
    request_id = Column(String(50), nullable=True)      # correlation_id da requisição


# Engine assíncrono para o banco de dados
engine = create_async_engine(CONFIG.database_url, echo=False)

# Factory de sessão assíncrona
SessionLocal = sessionmaker(
    bind=engine, class_=AsyncSession, expire_on_commit=False
)


async def init_db() -> None:
    """Cria as tabelas no banco de dados (idempotente — CREATE TABLE IF NOT EXISTS)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
