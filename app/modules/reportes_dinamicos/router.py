"""
Router del Motor Deep Learning de Reportes Inteligentes y Asistente de Datos.

Endpoints:
  - POST /api/ia/reportes/interpretar       → Interpretar consulta libre a plan de reporte
  - POST /api/ia/asistente-datos/preguntar   → Preguntar al asistente de datos
  - POST /api/ia/asistente-datos/planificar  → Generar plan de consulta
  - GET  /api/ia/reportes/catalogo           → Catálogo de entidades permitidas
  - GET  /api/ia/reportes/motor/status       → Estado del motor IA
  - POST /api/ia/reportes/transcribir        → Stub para transcripción de audio
  - POST /api/ia/reportes/respuesta-natural  → Generar respuesta natural con datos

El Motor IA solo interpreta. Nunca accede a bases de datos.
"""
from fastapi import APIRouter, HTTPException, Request
from typing import Optional, List, Dict, Any
import logging

from app.modules.reportes_dinamicos.schemas import (
    ReporteRequest, ReporteResponse, 
    AsistenteDatosRequest, AsistenteDatosResponse, PlanConsulta,
    TranscripcionRequest, CatalogoResponse, MotorTipo,
    AsistenciaExtendidaRequest, AsistenciaExtendidaResponse
)
from app.modules.reportes_dinamicos.prompt_builder import build_prompt_asistencia_extendida
from app.modules.reportes_dinamicos.interpretador_semantico import (
    interpretar_reporte, procesar_pregunta_asistente, generar_respuesta_natural
)
from app.modules.reportes_dinamicos.catalogo_reportes import get_catalogo_completo, get_catalogo_remoto, build_catalogo_from_remote
from app.modules.reportes_dinamicos.motor_ia_client import motor_ia_client
from app.modules.reportes_dinamicos.config import reportes_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ia/reportes", tags=["Reportes Inteligentes"])
router_asistente = APIRouter(prefix="/api/ia/asistente-datos", tags=["Asistente de Datos"])


# ============================================================
# REPORTES INTELIGENTES
# ============================================================

@router.post("/interpretar", response_model=ReporteResponse)
async def interpretar(req: ReporteRequest):
    """
    Interpreta una consulta en lenguaje natural y genera un plan de reporte estructurado.
    
    Flujo:
    1. Motor IA Avanzado (si está configurado)
    2. Motor Interno (modelo Keras propio)  
    3. Fallback controlado
    
    La IA solo genera el plan. El backend Spring Boot valida y ejecuta la consulta.
    """
    if not req.texto or not req.texto.strip():
        raise HTTPException(status_code=400, detail="El texto de la consulta no puede estar vacío.")
    
    logger.info(f"== NUEVA SOLICITUD DE INTERPRETACIÓN ==")
    logger.info(f"Texto recibido: '{req.texto}' (rol={req.rol}, usuario={req.usuarioId})")
    
    try:
        resultado = await interpretar_reporte(req.texto)
        logger.info(f"Resultado final -> motor: {resultado.motor}, intención: {resultado.intencionDetectada}, confianza: {resultado.confianza:.2f}")
        return resultado
    except Exception as e:
        logger.error(f"Error inesperado interpretando reporte: {e}")
        raise HTTPException(status_code=500, detail="Error interno del Motor IA al interpretar la solicitud.")


@router.post("/asistencia-extendida", response_model=AsistenciaExtendidaResponse)
async def asistencia_extendida(req: AsistenciaExtendidaRequest):
    """
    Genera una vista asistida (IA+) cuando el motor real de reportes falla o no tiene resultados.
    """
    logger.info("== SOLICITUD DE ASISTENCIA EXTENDIDA (IA+) Recibida ==")
    try:
        messages = build_prompt_asistencia_extendida(
            pregunta_original=req.preguntaOriginal,
            columnas_esperadas=req.columnasEsperadas,
            diagnostico=req.diagnosticoMotorReal or "No se pudo completar la consulta real.",
            datos_reales=req.datosRealesDisponibles or [],
            datos_previos=req.datosSimuladosPrevios or [],
            usuarios=req.usuariosReales or [],
            funcionarios=req.funcionariosReales or [],
            administradores=req.administradoresReales or [],
            politicas=req.politicasReales or [],
            departamentos=req.departamentosReales or [],
            estados=req.estadosReales or [],
            nodos=req.nombresNodosReales or []
        )
        resultado = await motor_ia_client.interpretar(messages)
        if not resultado:
            raise HTTPException(status_code=500, detail="El motor IA no devolvió ningún resultado.")
        
        columnas = resultado.get("columnas", req.columnasEsperadas)
        filas = resultado.get("filas", [])
        
        for fila in filas:
            if "_modo" not in fila:
                fila["_modo"] = "IA_PLUS"
            if "_origen" not in fila:
                fila["_origen"] = "asistido"
            if "_camposEstimados" not in fila:
                fila["_camposEstimados"] = [c for c in columnas if c not in ["politicaNombre", "responsableNombre", "usuarioNombre", "departamentoNombre", "funcionarioNombre", "nombre", "correo", "codigoTramite", "estadoInstancia", "estado"]]
                
        return AsistenciaExtendidaResponse(
            columnas=columnas,
            filas=filas,
            asistido=True
        )
    except Exception as e:
        logger.error(f"Error en asistencia extendida: {e}")
        raise HTTPException(status_code=500, detail=f"Error al generar asistencia extendida: {str(e)}")


@router.get("/catalogo", response_model=CatalogoResponse)
async def get_catalogo():
    """
    Retorna el catálogo completo de entidades, campos y operaciones permitidas.
    Solo se consultan entidades y campos de este catálogo.
    """
    catalogo = get_catalogo_completo()
    return CatalogoResponse(**catalogo)


@router.get("/motor/status")
async def motor_status():
    """
    Retorna el estado actual del Motor IA de Reportes Inteligentes.
    Indica qué motores están disponibles.
    """
    motor_avanzado_disponible = motor_ia_client.is_available()
    
    # Verificar motor interno
    motor_interno_disponible = False
    try:
        from pathlib import Path
        model_path = Path(__file__).resolve().parent / "models" / "modelo_reportes.keras"
        motor_interno_disponible = model_path.exists()
    except Exception:
        pass
    
    return {
        "motorIaAvanzado": {
            "disponible": motor_avanzado_disponible,
            "modelo": motor_ia_client.model if motor_avanzado_disponible else None
        },
        "motorInterno": {
            "disponible": motor_interno_disponible,
            "tipo": "Keras BiLSTM Multi-Output"
        },
        "motorFallback": {
            "disponible": True,
            "tipo": "Heurístico con catálogo estático"
        },
        "motorPrincipal": (
            "MOTOR_IA_AVANZADO" if motor_avanzado_disponible
            else "MOTOR_INTERNO" if motor_interno_disponible
            else "MOTOR_FALLBACK"
        )
    }

@router.get("/diagnostico")
async def get_diagnostico(request: Request):
    """
    Endpoint de diagnóstico para el microservicio IA.
    """
    remote_data = get_catalogo_remoto()
    _, _, _, origen = build_catalogo_from_remote(remote_data)
    
    motor_avanzado_disponible = motor_ia_client.is_available()
    api_key_presente = bool(reportes_settings.ia_api_key and len(reportes_settings.ia_api_key) > 5)
    
    motor_interno_disponible = False
    try:
        from pathlib import Path
        model_path = Path(__file__).resolve().parent / "models" / "modelo_reportes.keras"
        motor_interno_disponible = model_path.exists()
    except Exception:
        pass

    from app.modules.reportes_dinamicos.catalogo_reportes import BACKEND_REPORTES_CATALOGO_URL
    return {
        "servicio": "ia-reportes",
        "catalogoOrigen": origen,
        "catalogoDisponible": origen == "SPRING_BOOT",
        "motorAvanzadoHabilitado": reportes_settings.ia_enabled,
        "apiKeyPresente": api_key_presente,
        "modeloConfigurado": reportes_settings.ia_model,
        "backendCatalogoUrl": BACKEND_REPORTES_CATALOGO_URL,
        "fallbackLocalDisponible": True,
        "motorInternoDisponible": motor_interno_disponible
    }


@router.post("/transcribir")
async def transcribir_audio(req: TranscripcionRequest):
    """
    Stub para transcripción de audio.
    La transcripción real se implementa en el frontend con Web Speech API.
    """
    return {"textoTranscrito": "Transcripción no implementada en backend. Use Web Speech API en el navegador."}


@router.post("/respuesta-natural")
async def respuesta_natural(
    texto_original: str = "",
    datos: List[Dict[str, Any]] = []
):
    """
    Genera una respuesta en lenguaje natural basada en datos ya recuperados.
    Se usa después de que Spring Boot ejecuta la consulta y obtiene resultados.
    """
    resultado = await generar_respuesta_natural(texto_original, datos)
    return resultado


# ============================================================
# ASISTENTE DE DATOS
# ============================================================

@router_asistente.post("/preguntar")
async def preguntar_asistente(req: AsistenteDatosRequest):
    """
    Procesa una pregunta libre del usuario sobre los datos del sistema.
    Genera un plan de consulta que el backend Spring Boot debe ejecutar.
    
    La IA no accede a ninguna base de datos directamente.
    Solo genera un plan estructurado validado contra el catálogo.
    """
    if not req.texto or not req.texto.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    
    logger.info(f"Asistente de datos: '{req.texto[:100]}...' (rol={req.rol})")
    
    try:
        resultado = await procesar_pregunta_asistente(req.texto, req.contextoAdicional)
        return resultado
    except Exception as e:
        logger.error(f"Error en asistente de datos: {e}")
        raise HTTPException(status_code=500, detail="Error interno del Motor IA al procesar la pregunta.")


@router_asistente.post("/planificar")
async def planificar_consulta(req: AsistenteDatosRequest):
    """
    Genera un plan de consulta detallado sin ejecutarlo.
    El backend Spring Boot valida el plan y luego lo ejecuta si es válido.
    """
    if not req.texto or not req.texto.strip():
        raise HTTPException(status_code=400, detail="La pregunta no puede estar vacía.")
    
    resultado = await procesar_pregunta_asistente(req.texto, req.contextoAdicional)
    return resultado


@router_asistente.get("/catalogo")
async def catalogo_asistente():
    """Catálogo de fuentes de datos disponibles para el asistente."""
    return get_catalogo_completo()

import os
