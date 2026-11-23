from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="MEMCRAFT_", extra="ignore")

    database_url: str = "postgresql+psycopg://memcraft@127.0.0.1:5432/memcraft"
    shared_dir: Path = Path("data/shared")
    # None in the web container, which does not see this folder.
    private_dir: Path | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
