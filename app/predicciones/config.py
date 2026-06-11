from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent

class PrediccionesSettings(BaseSettings):
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    BACKEND_BASE_URL: str = "http://localhost:8080"
    BACKEND_DATASET_URL: str = "http://localhost:8080/api/deep-learning/dataset/rutas/export/csv"
    MODEL_DIR: str = str(BASE_DIR / "models")
    DATASET_DIR: str = str(BASE_DIR / "datasets")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = PrediccionesSettings()
