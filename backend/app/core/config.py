from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Dane bazy
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432

    YOLO_MODEL_PATH: str = "models_ai/best.pt"
    CNN_MODEL_PATH: str = "models_ai/cnn_48_v1.keras"

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