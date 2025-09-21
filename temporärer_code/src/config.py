from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    # Basispfade
    output_dir: Path = Path("output")
    upload_dir: Path = Path("uploads")
    temp_dir: Path = Path("temp")

    # E-Mail Konfiguration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = "pdfextractor@example.com"

    # API Konfiguration
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_extensions: list = [".pdf"]

    # Logging
    log_level: str = "INFO"
    log_file: Path = Path("logs/app.log")

    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()
