from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os

class Settings(BaseSettings):
    # Dane bazy (wymagane w .env)
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432

    # Ścieżki modeli - DODAJEMY ADNOTACJE TYPU str
    # Domyślne ścieżki ustawiamy tak, aby działały WEWNĄTRZ kontenera
    YOLO_MODEL_PATH: str = "models/best.pt"
    CNN_MODEL_PATH: str = "models/cnn_224_v3.keras"

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    # Konfiguracja ładowania
    model_config = SettingsConfigDict(
        env_file=".env", 
        extra="ignore",
        env_file_encoding="utf-8"
    )

settings = Settings()