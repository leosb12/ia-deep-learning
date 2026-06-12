from typing import Optional
from fastapi import APIRouter, Depends, Header

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
    x_ai_mode: Optional[str] = Header(None, alias="X-AI-Mode"),
    service: ClasificadorDinamicoService = Depends(obtener_clasificador_dinamico_service),
    modelo_local_service: ModeloPropioService = Depends(obtener_modelo_propio_service),
) -> RespuestaClasificacionDinamica:
    if x_ai_mode == "offline":
        # Helper mapping function from Keras labels to MongoDB policy IDs based on policy names
        def resolver_id_mongo(keras_label: str, politicas: list) -> str:
            mapeo = {
                "WIFI_INSTALACION": "Solicitar instalacion de internet WiFi",
                "WIFI_CAMBIO_PLAN": "Cambiar plan de internet",
                "WIFI_MEJORAR_VELOCIDAD": "Mejorar velocidad de internet",
                "WIFI_INTERNET_CAIDO": "Reportar internet caido",
                "WIFI_INTERNET_LENTO": "Reportar internet lento",
                "WIFI_CAMBIAR_PASSWORD": "Cambiar contrasena del WiFi",
                "WIFI_CAMBIAR_NOMBRE_RED": "Cambiar nombre de la red WiFi",
                "WIFI_VISITA_TECNICA": "Solicitar visita tecnica",
                "WIFI_REPROGRAMAR_VISITA": "Reprogramar visita tecnica",
                "WIFI_ESTADO_INSTALACION": "Consultar estado de instalacion",
                "WIFI_TRASLADO_SERVICIO": "Solicitar traslado de servicio",
                "WIFI_ACTUALIZAR_DATOS": "Actualizar datos del titular",
                "WIFI_CAMBIO_TITULAR": "Cambiar titular del servicio",
                "WIFI_CONSULTAR_FACTURA": "Consultar factura",
                "WIFI_RECLAMO_COBRO": "Reclamar cobro incorrecto",
                "WIFI_DUPLICADO_FACTURA": "Solicitar duplicado de factura",
                "WIFI_REGISTRAR_PAGO": "Registrar comprobante de pago",
                "WIFI_RECONEXION": "Solicitar reconexion de servicio",
                "WIFI_BAJA_SERVICIO": "Solicitar baja del servicio",
                "WIFI_SOPORTE_MODEM_ROUTER": "Solicitar soporte por modem o router",
                "WIFI_CAMBIO_EQUIPO": "Solicitar cambio de equipo"
            }
            nombre_esperado = mapeo.get(keras_label)
            if not nombre_esperado:
                return keras_label
            
            from app.modules.clasificador_solicitudes.service import normalizar_para_busqueda
            nombre_esperado_norm = normalizar_para_busqueda(nombre_esperado)
            for p in politicas:
                p_nombre_norm = normalizar_para_busqueda(p.nombre)
                if p_nombre_norm == nombre_esperado_norm or nombre_esperado_norm in p_nombre_norm or p_nombre_norm in nombre_esperado_norm:
                    return p.id
            return keras_label

        # 1. Run local Keras model classification
        resultado_local = modelo_local_service.clasificar(solicitud.texto)
        
        # 2. Enrich results using solicitud.politicas
        politicas_map = {p.id: p for p in solicitud.politicas}
        politica_id = resolver_id_mongo(resultado_local["politicaId"], solicitud.politicas)
        politica_match = politicas_map.get(politica_id)
        nombre_politica = politica_match.nombre if politica_match else None
        
        # Extract requirements locally using rules
        from app.modules.clasificador_solicitudes.service import extraer_requisitos_reglas
        requisitos_detectados = extraer_requisitos_reglas(solicitud.texto, solicitud.politicas)
        if solicitud.nombreDocumento:
            reqs_doc = extraer_requisitos_reglas(solicitud.nombreDocumento, solicitud.politicas)
            requisitos_detectados = list(set(requisitos_detectados).union(reqs_doc))
        
        coincidentes = []
        faltantes = []
        if politica_match:
            reqs_politica = [r.nombre for r in politica_match.requisitosIniciales]
            coincidentes = [r for r in reqs_politica if r in requisitos_detectados]
            faltantes = [r for r in reqs_politica if r not in requisitos_detectados]
            
        top_resultados_enriquecidos = []
        for r in resultado_local["topResultados"]:
            keras_lbl = r["politicaId"]
            p_id = resolver_id_mongo(keras_lbl, solicitud.politicas)
            p_match = politicas_map.get(p_id)
            reqs_iniciales = [req.nombre for req in p_match.requisitosIniciales] if p_match else []
            
            top_resultados_enriquecidos.append({
                "politicaId": p_id,
                "nombrePolitica": p_match.nombre if p_match else keras_lbl,
                "confianza": r["confianza"],
                "scoreRequisitos": 0.0,
                "scoreSemantico": r["confianza"],
                "scoreFinal": r["confianza"],
                "requisitosCoincidentes": [],
                "requisitosFaltantes": reqs_iniciales,
            })
            
        return RespuestaClasificacionDinamica(
            politicaId=politica_id,
            nombrePolitica=nombre_politica,
            confianza=resultado_local["confianza"],
            origen=resultado_local["origen"],
            metodoRecomendacion="MODELO_PROPIO_LOCAL",
            requiereMasInformacion=len(faltantes) > 0 if politica_match else False,
            requisitosDetectados=requisitos_detectados,
            requisitosCoincidentes=coincidentes,
            requisitosFaltantes=faltantes,
            topResultados=top_resultados_enriquecidos,
            analisisDeepSeek=None
        )

    resultado = service.clasificar(
        solicitud.texto,
        solicitud.politicas,
        solicitud.usarDeepSeek,
        solicitud.nombreDocumento,
        usar_solo_requisitos_iniciales=solicitud.usarSoloRequisitosIniciales,
    )
    return RespuestaClasificacionDinamica(**resultado)
