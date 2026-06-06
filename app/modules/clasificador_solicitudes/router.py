from fastapi import APIRouter, Depends

from app.modules.clasificador_solicitudes.schemas import (
    RespuestaClasificacion,
    RespuestaClasificacionDinamica,
    SolicitudClasificacion,
    SolicitudClasificacionDinamica,
)
from app.modules.clasificador_solicitudes.service import (
    ClasificadorDinamicoService,
    ModeloPropioService,
    clasificador_dinamico_service,
    modelo_propio_service,
)

router = APIRouter(tags=["clasificador-solicitudes"])


def obtener_modelo_propio_service() -> ModeloPropioService:
    return modelo_propio_service


def obtener_clasificador_dinamico_service() -> ClasificadorDinamicoService:
    return clasificador_dinamico_service


@router.post("/api/ia/modelo-propio/clasificar", response_model=RespuestaClasificacion)
async def clasificar_solicitud_compatibilidad(
    solicitud: SolicitudClasificacion,
    service: ModeloPropioService = Depends(obtener_modelo_propio_service),
) -> RespuestaClasificacion:
    return await clasificar_solicitud(solicitud, service)


@router.post("/api/deep-learning/clasificador-solicitudes/clasificar", response_model=RespuestaClasificacion)
async def clasificar_solicitud(
    solicitud: SolicitudClasificacion,
    service: ModeloPropioService = Depends(obtener_modelo_propio_service),
) -> RespuestaClasificacion:
    resultado = service.clasificar(solicitud.texto)
    return RespuestaClasificacion(**resultado)


@router.post(
    "/api/deep-learning/clasificador-solicitudes/clasificar-dinamico",
    response_model=RespuestaClasificacionDinamica,
    response_model_exclude_none=True,
)
async def clasificar_solicitud_dinamica(
    solicitud: SolicitudClasificacionDinamica,
    service: ClasificadorDinamicoService = Depends(obtener_clasificador_dinamico_service),
) -> RespuestaClasificacionDinamica:
    resultado = service.clasificar(solicitud.texto, solicitud.politicas, solicitud.usarDeepSeek)
    return RespuestaClasificacionDinamica(**resultado)
