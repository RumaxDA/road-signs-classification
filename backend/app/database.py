from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from app.core.config import settings

engine = create_async_engine(settings.ASYNC_DATABASE_URL, echo=True)
async_session = async_sessionmaker(engine, expire_on_commit=False)