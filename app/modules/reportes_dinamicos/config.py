"""
Configuración del módulo de Reportes Inteligentes / Motor Deep Learning.
Reutiliza las variables de entorno existentes del proveedor IA configurado.
"""
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


class ReportesSettings(BaseModel):
    """
    Configuración del Motor IA para Reportes Inteligentes.
    Internamente reutiliza las variables DEEPSEEK_* ya configuradas.
    """
    # Proveedor IA (interno, reutiliza config existente)
    ia_api_key: str | None = os.getenv("DEEPSEEK_API_KEY")
    ia_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    ia_model: str = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
    ia_enabled: bool = _env_bool("DEEPSEEK_ENABLED", False)
    ia_timeout_seconds: float = _env_float("DEEPSEEK_TIMEOUT_SECONDS", 30.0)
    ia_temperature: float = _env_float("DEEPSEEK_TEMPERATURE", 0.2)
    ia_max_tokens: int = _env_int("DEEPSEEK_MAX_TOKENS", 4096)

    # Límites de seguridad
    max_resultados_preview: int = 500
    max_resultados_exportacion: int = 5000
    limite_default: int = 50
    confianza_minima: float = 0.5

    # Motor interno (Keras)
    motor_interno_habilitado: bool = True


reportes_settings = ReportesSettings()
