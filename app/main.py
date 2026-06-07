from fastapi import FastAPI

from app.modules.clasificador_solicitudes.router import router as router_clasificador_solicitudes
from app.shared.config import settings
from app.shared.health_router import router as health_router
from app.predicciones.router import router as router_predicciones


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.service_name,
        version="1.0.0",
        description="Servicio de modelos deep learning propios para casos de uso de IA.",
    )
    app.include_router(health_router)
    app.include_router(router_clasificador_solicitudes)
    app.include_router(router_predicciones, prefix="/api/predicciones", tags=["Predicciones"])

    return app


app = create_app()
