import os
import logging
from pathlib import Path
from typing import Optional
from datetime import datetime


def setup_logging(
    log_level: str = "INFO", log_file: Optional[Path] = None
) -> logging.Logger:
    """Richtet das Logging-System ein"""
    logger = logging.getLogger(__name__)
    logger.setLevel(getattr(logging, log_level.upper()))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def validate_file_path(file_path: str, allowed_extensions: list = None) -> bool:
    """Validiert eine Datei anhand ihrer Erweiterung"""
    if not allowed_extensions:
        return True

    file_extension = Path(file_path).suffix.lower()
    return file_extension in allowed_extensions


def get_file_size(file_path: str) -> int:
    """Gibt die Dateigröße in Bytes zurück"""
    return os.path.getsize(file_path)


def format_file_size(size_bytes: int) -> str:
    """Formatiert die Dateigröße in menschenlesbarem Format"""
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.2f}{size_names[i]}"


def create_unique_filename(original_name: str, prefix: str = "") -> str:
    """Erstellt einen eindeutigen Dateinamen mit Zeitstempel"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name_without_ext = Path(original_name).stem
    extension = Path(original_name).suffix

    return f"{prefix}{name_without_ext}_{timestamp}{extension}"


def ensure_directory_exists(directory_path: str) -> Path:
    """Stellt sicher, dass ein Verzeichnis existiert"""
    path = Path(directory_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clean_temp_files(directory_path: str, max_age_hours: int = 24) -> None:
    """Löscht alte temporäre Dateien"""
    temp_dir = Path(directory_path)
    if not temp_dir.exists():
        return

    current_time = datetime.now()
    for file_path in temp_dir.glob("*"):
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            age_hours = (current_time - file_time).total_seconds() / 3600

            if age_hours > max_age_hours:
                try:
                    file_path.unlink()
                except Exception as e:
                    logging.warning(
                        f"Konnte temporäre Datei nicht löschen: {file_path}, Fehler: {e}"
                    )
