"""Application settings for the skating biomechanics web API and worker."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central settings, loaded from environment variables or .env file."""

    valkey_host: str = "localhost"
    valkey_port: int = 6379
    valkey_db: int = 0
    valkey_password: str | None = None

    outputs_dir: str = "data/uploads"
    worker_max_jobs: int = 1
    worker_retry_delays: list[int] = [30, 120]
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    log_level: str = "INFO"
    task_ttl_seconds: int = 24 * 60 * 60

    def build_valkey_url(self) -> str:
        auth = f":{self.valkey_password}@" if self.valkey_password else ""
        return f"redis://{auth}{self.valkey_host}:{self.valkey_port}/{self.valkey_db}"

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",
    }


def get_settings() -> Settings:
    if get_settings._instance is None:  # type: ignore[attr-defined]
        get_settings._instance = Settings()  # type: ignore[attr-defined]
    return get_settings._instance  # type: ignore[attr-defined]


get_settings._instance = None  # type: ignore[attr-defined]
