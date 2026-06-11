from fastapi import FastAPI

from app.modules.clasificador_solicitudes.router import router as router_clasificador_solicitudes
from app.shared.config import settings
from app.shared.health_router import router as health_router
from app.modules.predicciones.router import router as router_predicciones


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.service_name,
        version="2.0.0",
        description="Servicio de modelos deep learning propios para casos de uso de IA: predicciones, reportes inteligentes y asistente de datos.",
    )
    app.include_router(health_router)
    app.include_router(router_clasificador_solicitudes)
    app.include_router(router_predicciones, prefix="/api/predicciones", tags=["Predicciones"])

    # Motor Deep Learning - Reportes Inteligentes
    from app.modules.reportes_dinamicos.router import router as router_reportes
    from app.modules.reportes_dinamicos.router import router_asistente
    from app.modules.reportes_visuales.routes import router as router_reportes_visuales
    app.include_router(router_reportes)
    app.include_router(router_asistente)
    app.include_router(router_reportes_visuales)

    @app.on_event("startup")
    async def startup_event():
        import asyncio
        from app.modules.predicciones.services.startup_training import auto_train_if_missing
        asyncio.create_task(auto_train_if_missing())

    return app


app = create_app()
