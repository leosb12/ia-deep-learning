import os

from pydantic import BaseModel

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


class Settings(BaseModel):
    service_name: str = "ia-deep-learning-service"
    suggested_port: int = 8010
    deepseek_enabled: bool = _env_bool("DEEPSEEK_ENABLED", False)
    deepseek_auto_analysis: bool = _env_bool("DEEPSEEK_AUTO_ANALYSIS", False)
    deepseek_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    deepseek_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
    deepseek_timeout_seconds: float = _env_float("DEEPSEEK_TIMEOUT_SECONDS", 8.0)
    deepseek_max_policies: int = _env_int("DEEPSEEK_MAX_POLICIES", 8)
    deepseek_auto_confidence_threshold: float = _env_float("DEEPSEEK_AUTO_CONFIDENCE_THRESHOLD", 0.65)

    # URL del backend — en Docker Compose usar nombre de servicio: http://backend:8080
    # En local: http://localhost:8080
    backend_base_url: str = os.getenv("BACKEND_BASE_URL", "http://localhost:8080")
    backend_dataset_url: str = os.getenv(
        "BACKEND_DATASET_URL",
        os.getenv("BACKEND_BASE_URL", "http://localhost:8080") + "/api/deep-learning/dataset/rutas/export/csv",
    )

    model_dir: str = os.getenv("MODEL_DIR", "app/modules/predicciones/models")
    dataset_dir: str = os.getenv("DATASET_DIR", "app/modules/predicciones/datasets")


settings = Settings()
