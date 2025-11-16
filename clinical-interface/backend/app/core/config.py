"""
Clinical Interface Configuration
Settings for the stenosis detection clinical interface backend
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Coronary Stenosis Detection System"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "Clinical interface for AI-powered coronary artery stenosis detection"

    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    RELOAD: bool = False

    # CORS Settings
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
    ]

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./stenosis_clinical.db"

    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # File Storage
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = DATA_DIR / "uploads"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    REPORTS_DIR: Path = DATA_DIR / "reports"

    # Model Configuration
    MODEL_DIR: Path = BASE_DIR.parent.parent / "models"
    SEGMENTATION_MODEL_PATH: Optional[str] = None
    CLASSIFICATION_MODEL_PATH: Optional[str] = None

    # Processing Configuration
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_EXTENSIONS: set[str] = {".nii", ".nii.gz"}

    # Preprocessing Parameters
    HU_WINDOW_MIN: int = -100
    HU_WINDOW_MAX: int = 1000
    TARGET_SPACING: float = 0.8  # mm, isotropic
    NORMALIZE_MIN: float = 0.0
    NORMALIZE_MAX: float = 1.0

    # Inference Configuration
    DEVICE: str = "cuda"  # or "cpu"
    BATCH_SIZE: int = 1
    ENABLE_MIXED_PRECISION: bool = True

    # Stenosis Classification Thresholds
    STENOSIS_NORMAL_MAX: float = 0.25
    STENOSIS_MILD_MAX: float = 0.50
    STENOSIS_MODERATE_MAX: float = 0.70
    # > 0.70 is severe

    # Security
    SECRET_KEY: str = "CHANGE_THIS_IN_PRODUCTION_USE_openssl_rand_hex_32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24  # 24 hours

    # Job Queue
    CELERY_TASK_TIME_LIMIT: int = 1800  # 30 minutes
    CELERY_TASK_SOFT_TIME_LIMIT: int = 1500  # 25 minutes

    class Config:
        env_file = ".env"
        case_sensitive = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings (dependency injection)"""
    return settings
