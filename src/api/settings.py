"""Application settings and SQLite engine construction."""

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from sqlalchemy import Engine, URL, event
from sqlmodel import create_engine

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PLANETARY_API_", env_file=WORKSPACE_ROOT / ".env",
        env_file_encoding="utf-8", extra="ignore",
    )

    app_name: str = "Model Evaluator"
    database_path: Path = WORKSPACE_ROOT / ".state" / "planetary.sqlite3"
    sql_echo: bool = False
    max_upload_bytes: int = 256 * 1024 * 1024
    max_unpacked_bytes: int = 1024 * 1024 * 1024
    max_archive_files: int = 10000
    download_timeout: float = 30
    worker_poll_seconds: float = 2
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    @field_validator("database_path")
    @classmethod
    def resolve_database_path(cls, value: Path) -> Path:
        path = (WORKSPACE_ROOT / value).resolve() if not value.is_absolute() else value.resolve()
        if "data_rover" in {part.lower() for part in path.parts}:
            raise ValueError("The protected archive cannot contain the database")
        return path

    @property
    def database_url(self) -> URL:
        return URL.create("sqlite+pysqlite", database=str(self.database_path))

    def create_engine(self) -> Engine:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        engine = create_engine(
            self.database_url, echo=self.sql_echo,
            connect_args={"check_same_thread": False, "timeout": 30},
        )

        @event.listens_for(engine, "connect")
        def configure_sqlite(connection, _):
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()

        return engine


def orbital_path(value: str) -> Path:
    """Resolve an orbital configuration path without opening any data."""
    path = (WORKSPACE_ROOT / value).resolve()
    if not path.is_relative_to(WORKSPACE_ROOT / "data_orbital"):
        raise ValueError("Dataset paths must be inside data_orbital/")
    return path
