from pydantic_settings import BaseSettings

class PrediccionesSettings(BaseSettings):
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"
    BACKEND_BASE_URL: str = "http://localhost:8080"
    BACKEND_DATASET_URL: str = "http://localhost:8080/api/deep-learning/dataset/rutas/export/csv"
    MODEL_DIR: str = "app/predicciones/models"
    DATASET_DIR: str = "app/predicciones/datasets"

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = PrediccionesSettings()
