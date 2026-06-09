"""
Interpretador Semántico - Motor Deep Learning de Reportes Inteligentes.

Orquesta la interpretación de consultas libres usando:
1. Motor IA Avanzado (proveedor API) - flujo principal si está configurado
2. Motor Interno (modelo Keras propio) - fallback y demostración académica
3. Motor Fallback - respuesta controlada cuando nada más está disponible

La IA solo interpreta. Nunca accede a bases de datos.
"""
import logging
import json
import numpy as np
import pickle
from pathlib import Path
from typing import Optional, Dict, Any, List

from app.modules.reportes_dinamicos.schemas import (
    ReporteResponse, Metrica, Filtro, Ordenamiento, MotorTipo,
    PlanConsulta, AsistenteDatosResponse
)
from app.modules.reportes_dinamicos.motor_ia_client import motor_ia_client
from app.modules.reportes_dinamicos.prompt_builder import (
    build_prompt_reportes, build_prompt_asistente, build_prompt_respuesta_final
)
from app.modules.reportes_dinamicos.catalogo_reportes import validar_plan_consulta

import unicodedata
import re

def normalize_text(text: str) -> str:
    # Convert to lowercase
    text = text.lower()
    # Normalize unicode to decompose accents
    text = "".join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    # Replace smart quotes with standard quotes if any
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    # Clean extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def has_entity_and_condition_clear(normalized: str) -> bool:
    has_ent = any(k in normalized for k in ["tramite", "solicitud", "instancia", "politica", "usuario", "notificacion", "documento", "archivo", "pago", "tarea"])
    has_cond = any(k in normalized for k in ["en curso", "activo", "abierto", "finalizado", "completado", "rechazado", "pendiente", "por estado", "mas", "menos", "sin leer", "no leid"])
    return has_ent and has_cond

def aplicar_reglas_semanticas_basicas(texto: str) -> Optional[ReporteResponse]:
    normalized = normalize_text(texto)
    
    has_tramite_or_solicitud = any(k in normalized for k in ["tramite", "solicitud", "instancia"])
    
    # ============================================================
    # CASO CUELLOS DE BOTELLA — MÁXIMA PRIORIDAD
    # Detectar ANTES que cualquier otra regla para evitar que la IA
    # intente resolver cuelloBotella.tipo en predicciones_ia.
    # Spring Boot usa AnalyticsService.getBottlenecks() directamente.
    # ============================================================
    has_cuello = any(k in normalized for k in [
        "cuello de botella", "cuellos de botella", "bottleneck",
        "proceso trabado", "procesos trabados",
        "nodo.*demora", "demora.*nodo", "etapa.*lenta", "etapas lentas",
        "nodo lento", "nodos lentos", "nodo.*lento", "lento.*nodo",
        "funcionario saturado", "funcionarios saturados",
        "departamento saturado", "acumulacion de tareas",
        "mas demora", "mayor demora", "generando demora",
        "recomendacion.*ia.*solucionarlo", "ia.*recomienda.*solucionarlo"
    ])
    # Detección más amplia: "qué nodos/etapas generan demora" + "recomendación IA"
    if not has_cuello:
        has_demora = any(k in normalized for k in ["demora", "retraso", "tardando", "tarda"])
        has_rec_ia = any(k in normalized for k in ["recomendacion", "recomienda", "ia", "solucionarlo", "bottleneck"])
        has_cuello = has_demora and has_rec_ia
    # Detectar también: "qué etapas/nodos del workflow generan más demora"
    if not has_cuello:
        has_etapa_nodo = any(k in normalized for k in ["etapa", "nodo", "fase", "paso", "actividad"])
        has_generan_demora = any(k in normalized for k in ["generando", "generan", "causa", "causan", "mas demora", "mayor demora", "mas retraso"])
        has_cuello = has_etapa_nodo and has_generan_demora

    if has_cuello:
        return ReporteResponse(
            titulo="Cuellos de Botella del Sistema",
            descripcion="Análisis de cuellos de botella usando IA real. Muestra tipo, nombre, severidad, evidencia, impacto y recomendación.",
            intencionDetectada="cuellos_botella",
            entidadPrincipal="analiticas",
            campos=["tipo", "nombre", "severidad", "evidencia", "impacto", "recomendacion"],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.98,
            motor=MotorTipo.MOTOR_FALLBACK
        )

    # G) Política más utilizada / más usadas este mes
    has_politica = any(k in normalized for k in ["politica"])
    has_most_used = any(k in normalized for k in ["mas utilizada", "mas usadas", "mas usada", "mas recurrentes", "mas recurrente", "mas comunes", "mas comun"])
    if has_politica and has_most_used:
        # Detectar si pide estado de la política
        pide_estado_politica = any(k in normalized for k in ["estado de la politica", "estado de politica", "estado del workflow", "estado politica"])
        # Detectar si pide cantidad de trámites
        pide_cantidad = any(k in normalized for k in ["cantidad de tramites", "tramites iniciados", "cantidad tramites", "tramites"])
        # Detectar filtro temporal
        filtros_tiempo = []
        if "este mes" in normalized or "mes actual" in normalized:
            filtros_tiempo = [Filtro(campo="fechaCreacion", operador="mes_actual", valor=None)]
        elif "este ano" in normalized or "este año" in normalized:
            filtros_tiempo = [Filtro(campo="fechaCreacion", operador="anio_actual", valor=None)]
        
        # Determinar campos de salida
        agrupaciones_out = ["politicaNombre"]
        campos_out = ["politicaNombre"]
        if pide_estado_politica:
            agrupaciones_out.append("politicaEstado")
            campos_out.append("politicaEstado")
        campos_out.append("cantidadTramites")
        
        return ReporteResponse(
            titulo="Políticas más usadas",
            descripcion="Políticas con mayor cantidad de trámites iniciados.",
            intencionDetectada="politica_mas_utilizada",
            entidadPrincipal="instancias_politica",
            campos=campos_out,
            metricas=[Metrica(campo="id", operacion="count", alias="cantidadTramites")],
            agrupaciones=agrupaciones_out,
            filtros=filtros_tiempo,
            ordenamiento=[Ordenamiento(campo="cantidadTramites", direccion="desc")],
            limite=10,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.95,
            motor=MotorTipo.MOTOR_FALLBACK
        )

    # ============================================================
    # CASO DINERO / MONTO / PAGOS — ALTA PRIORIDAD
    # Si el usuario pregunta por dinero, monto, recaudación, ingresos
    # la entidad base debe ser "pagos", no "instancias_politica".
    # ============================================================
    has_dinero = any(k in normalized for k in [
        "dinero", "monto total", "cuanto se genero", "cuanto genero",
        "recaudado", "recaudacion", "ingreso", "ingresos", "cobrado",
        "cobro", "cuanto dinero", "dinero generado", "plata generada",
        "generado por politica", "ganado por politica"
    ])
    if has_dinero:
        # Detectar si también pide estado de pago y nombre de política
        has_estado_pago = any(k in normalized for k in ["estado de pago", "estado del pago", "estadopago"])
        has_nombre_politica = any(k in normalized for k in ["nombre de la politica", "politica", "por politica"])
        
        agrupaciones = []
        campos = []
        if has_nombre_politica:
            agrupaciones.append("politicaNombre")
            campos.append("politicaNombre")
        if has_estado_pago:
            agrupaciones.append("estado")
            campos.append("estado")
        
        campos_salida = list(agrupaciones) + ["montoTotal", "cantidadPagos"]
        
        return ReporteResponse(
            titulo="Dinero Generado por Política",
            descripcion="Reporte de montos y pagos agrupados por política.",
            intencionDetectada="reporte_pagos_por_politica",
            entidadPrincipal="pagos",
            campos=campos_salida if campos_salida else ["montoTotal", "cantidadPagos"],
            metricas=[
                Metrica(campo="monto", operacion="sum", alias="montoTotal"),
                Metrica(campo="id", operacion="count", alias="cantidadPagos")
            ],
            agrupaciones=agrupaciones if agrupaciones else [],
            ordenamiento=[Ordenamiento(campo="montoTotal", direccion="desc")],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.97,
            motor=MotorTipo.MOTOR_FALLBACK
        )

    # CASO A: Tareas de una política
    has_tareas = any(k in normalized for k in ["tarea", "nodo", "paso"])
    has_politica = any(k in normalized for k in ["politica", "workflow", "flujo"])
    has_tiene = any(k in normalized for k in ["tiene", "del", "de la", "pertenece"])
    if has_tareas and has_politica and has_tiene and "pendiente" not in normalized and "funcionario" not in normalized:
        return ReporteResponse(
            titulo="Nodos de Política",
            descripcion="Nodos/Tareas internas que componen una política o workflow.",
            intencionDetectada="listar_nodos_politica",
            entidadPrincipal="politicas_negocio",
            campos=["nombre", "nodos.nombre", "nodos.tipo", "nodos.rol"],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.98,
            motor=MotorTipo.MOTOR_FALLBACK
        )

    # CASO B: Políticas con nodo específico
    if has_politica and any(k in normalized for k in ["aprobacion", "pago", "notificacion"]) and has_tareas:
        return ReporteResponse(
            titulo="Políticas por tipo de nodo",
            descripcion="Políticas que contienen nodos específicos.",
            intencionDetectada="politicas_por_nodo",
            entidadPrincipal="politicas_negocio",
            campos=["nombre", "nodos.nombre", "nodos.tipo"],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.95,
            motor=MotorTipo.MOTOR_FALLBACK
        )

    # CASO C: Tareas pendientes o asignadas (tareas_actividad)
    has_tareas_pend = any(k in normalized for k in ["tarea", "trabajo pendiente", "actividad"])
    has_funcionario = any(k in normalized for k in ["funcionario", "usuario", "asignada", "soporte", "tiene mas"])
    has_pendiente = any(k in normalized for k in ["pendiente", "en curso", "tiene mas"])
    if has_tareas_pend and (has_funcionario or has_pendiente):
        if "mas" in normalized or "mayor" in normalized:
            return ReporteResponse(
                titulo="Funcionario con más tareas pendientes",
                descripcion="Funcionario con la mayor cantidad de tareas asignadas pendientes.",
                intencionDetectada="funcionario_mas_tareas",
                entidadPrincipal="tareas_actividad",
                campos=["responsableId", "cantidad"],
                metricas=[Metrica(campo="id", operacion="count", alias="cantidad")],
                agrupaciones=["responsableId"],
                filtros=[Filtro(campo="estado", operador="=", valor="PENDIENTE")],
                ordenamiento=[Ordenamiento(campo="cantidad", direccion="desc")],
                limite=1,
                formatoSalida="pantalla",
                visualizacion="tabla",
                requiereAclaracion=False,
                confianza=0.98,
                motor=MotorTipo.MOTOR_FALLBACK
            )
        else:
            return ReporteResponse(
                titulo="Tareas pendientes asignadas",
                descripcion="Listado de tareas reales asignadas a funcionarios.",
                intencionDetectada="listar_tareas_asignadas",
                entidadPrincipal="tareas_actividad",
                campos=["actividadNombre", "responsableId", "estado", "fechaCreacion", "instanciaId"],
                filtros=[Filtro(campo="estado", operador="=", valor="PENDIENTE")],
                limite=50,
                formatoSalida="pantalla",
                visualizacion="tabla",
                requiereAclaracion=False,
                confianza=0.98,
                motor=MotorTipo.MOTOR_FALLBACK
            )

    # CASO D: Políticas creadas
    if any(k in normalized for k in ["politicas creadas", "workflows configurados", "flujos existen", "tramites disponibles"]):
        return ReporteResponse(
            titulo="Políticas de Negocio Configuradas",
            descripcion="Listado de políticas o workflows creados en el sistema.",
            intencionDetectada="listar_politicas",
            entidadPrincipal="politicas_negocio",
            campos=["nombre", "estado", "fechaCreacion", "requierePago"],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.98,
            motor=MotorTipo.MOTOR_FALLBACK
        )

    # CASO E: Políticas iniciadas
    if any(k in normalized for k in ["politicas iniciadas", "workflows iniciados", "tramites iniciados", "solicitudes iniciadas"]):
        return ReporteResponse(
            titulo="Trámites/Políticas Iniciadas",
            descripcion="Listado de ejecuciones reales de políticas.",
            intencionDetectada="listar_instancias",
            entidadPrincipal="instancias_politica",
            campos=["codigoTramite", "politicaId", "estadoInstancia", "fechaCreacion", "creadaPor"],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.98,
            motor=MotorTipo.MOTOR_FALLBACK
        )
    
    # A) Trámites en curso
    is_active_or_in_progress = any(k in normalized for k in ["en curso", "activo", "abierto", "en proceso"])
    if has_tramite_or_solicitud and is_active_or_in_progress:
        return ReporteResponse(
            titulo="Trámites en curso",
            descripcion="Listado de trámites actualmente en curso.",
            intencionDetectada="listar_tramites_por_estado",
            entidadPrincipal="instancias_politica",
            campos=["codigoTramite", "estadoInstancia", "fechaCreacion", "creadaPor", "politicaId"],
            filtros=[Filtro(campo="estadoInstancia", operador="=", valor="EN_CURSO")],
            ordenamiento=[Ordenamiento(campo="fechaCreacion", direccion="desc")],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.95,
            motor=MotorTipo.MOTOR_FALLBACK
        )
        
    # B) Trámites finalizados
    is_finished = any(k in normalized for k in ["finalizado", "completado", "terminado", "cerrado"])
    if has_tramite_or_solicitud and is_finished:
        return ReporteResponse(
            titulo="Trámites finalizados",
            descripcion="Listado de trámites finalizados.",
            intencionDetectada="listar_tramites_por_estado",
            entidadPrincipal="instancias_politica",
            campos=["codigoTramite", "estadoInstancia", "fechaCreacion", "creadaPor", "politicaId"],
            filtros=[Filtro(campo="estadoInstancia", operador="=", valor="FINALIZADA")],
            ordenamiento=[Ordenamiento(campo="fechaCreacion", direccion="desc")],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.95,
            motor=MotorTipo.MOTOR_FALLBACK
        )
        
    # C) Trámites rechazados
    is_rejected = any(k in normalized for k in ["rechazado", "denegado", "no aprobado"])
    if has_tramite_or_solicitud and is_rejected:
        return ReporteResponse(
            titulo="Trámites rechazados",
            descripcion="Listado de trámites rechazados.",
            intencionDetectada="listar_tramites_por_estado",
            entidadPrincipal="instancias_politica",
            campos=["codigoTramite", "estadoInstancia", "fechaCreacion", "creadaPor", "politicaId"],
            filtros=[Filtro(campo="estadoInstancia", operador="=", valor="RECHAZADA")],
            ordenamiento=[Ordenamiento(campo="fechaCreacion", direccion="desc")],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.95,
            motor=MotorTipo.MOTOR_FALLBACK
        )
        
    # D) Trámites pendientes de pago
    is_pending_payment = any(k in normalized for k in ["pendiente de pago", "pendientes de pago", "falta pagar", "sin pagar"])
    if has_tramite_or_solicitud and is_pending_payment:
        return ReporteResponse(
            titulo="Trámites pendientes de pago",
            descripcion="Listado de trámites pendientes de pago.",
            intencionDetectada="listar_tramites_por_estado",
            entidadPrincipal="instancias_politica",
            campos=["codigoTramite", "estadoInstancia", "fechaCreacion", "creadaPor", "politicaId"],
            filtros=[Filtro(campo="estadoInstancia", operador="=", valor="PENDIENTE_PAGO")],
            ordenamiento=[Ordenamiento(campo="fechaCreacion", direccion="desc")],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.95,
            motor=MotorTipo.MOTOR_FALLBACK
        )

    # E) Conteo por estado
    has_quantity = any(k in normalized for k in ["cuanto", "cantidad", "conteo", "numero"])
    has_by_status = any(k in normalized for k in ["por estado", "agrupados por estado", "segun el estado", "segun estado"])
    if has_tramite_or_solicitud and has_quantity and has_by_status:
        return ReporteResponse(
            titulo="Trámites por estado",
            descripcion="Conteo de trámites agrupados por estado.",
            intencionDetectada="conteo_por_estado",
            entidadPrincipal="instancias_politica",
            campos=["estadoInstancia", "cantidad"],
            metricas=[Metrica(campo="id", operacion="count", alias="cantidad")],
            agrupaciones=["estadoInstancia"],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.95,
            motor=MotorTipo.MOTOR_FALLBACK
        )
        
    # F) Usuario con más trámites
    has_user = any(k in normalized for k in ["usuario", "cliente", "persona", "quien"])
    has_most_tramites = any(k in normalized for k in ["inicio mas", "creo mas", "tiene mas", "mas tramites", "mayor cantidad"])
    if has_tramite_or_solicitud and has_user and has_most_tramites:
        return ReporteResponse(
            titulo="Usuario con más trámites",
            descripcion="Usuario que ha iniciado mayor cantidad de trámites.",
            intencionDetectada="usuario_mas_tramites",
            entidadPrincipal="instancias_politica",
            campos=["creadaPor", "cantidad"],
            metricas=[Metrica(campo="id", operacion="count", alias="cantidad")],
            agrupaciones=["creadaPor"],
            ordenamiento=[Ordenamiento(campo="cantidad", direccion="desc")],
            limite=1,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.95,
            motor=MotorTipo.MOTOR_FALLBACK
        )
        
    # G) Política más utilizada — ya cubierto en la sección mejorada al inicio de la función

    # H) Documentos subidos
    has_document = any(k in normalized for k in ["documento", "archivo"])
    has_uploaded = any(k in normalized for k in ["subio", "subieron", "cargaron", "cargo", "subir"])
    has_most = any(k in normalized for k in ["quien", "mas", "mayor"])
    if has_document and has_uploaded and has_most:
        catalogo = get_catalogo_completo()
        entidad = "archivos_adjuntos" if "archivos_adjuntos" in catalogo["mongo"] else "metadatos_archivos"
        campos = catalogo["mongo"].get(entidad, []) or catalogo["dynamo"].get(entidad, [])
        agrupacion_campo = "subidoPor" if entidad == "archivos_adjuntos" else "usuarioSubio"
        metric_campo = "id" if entidad == "archivos_adjuntos" else "archivoId"
        
        # Check if date field exists in catalog for documents
        date_field = None
        for f in ["fecha", "fechaCreacion", "fechaSubida", "fecha_creacion"]:
            if f in campos:
                date_field = f
                break
                
        if date_field:
            return ReporteResponse(
                titulo="Documentos subidos por usuario este mes",
                descripcion="Usuario que ha subido más documentos este mes.",
                intencionDetectada="usuario_mas_documentos",
                entidadPrincipal=entidad,
                campos=[agrupacion_campo, "cantidad"],
                metricas=[Metrica(campo=metric_campo, operacion="count", alias="cantidad")],
                agrupaciones=[agrupacion_campo],
                filtros=[Filtro(campo=date_field, operador="mes_actual", valor=None)],
                ordenamiento=[Ordenamiento(campo="cantidad", direccion="desc")],
                limite=1,
                formatoSalida="pantalla",
                visualizacion="tabla",
                requiereAclaracion=False,
                confianza=0.95,
                motor=MotorTipo.MOTOR_FALLBACK
            )
        else:
            # Pedir aclaracion o devolver advertencia controlada
            return ReporteResponse(
                titulo="Documentos subidos por usuario",
                descripcion="La entidad de documentos no tiene un campo de fecha para filtrar por este mes.",
                intencionDetectada="usuario_mas_documentos",
                entidadPrincipal=entidad,
                requiereAclaracion=True,
                preguntaAclaratoria="La entidad de archivos no tiene un campo de fecha en el catálogo. ¿Deseas ver el ranking histórico de quién subió más documentos?",
                confianza=0.95,
                motor=MotorTipo.MOTOR_FALLBACK
            )
        
    # I) Notificaciones sin leer
    has_notification = any(k in normalized for k in ["notificacion", "alerta"])
    has_unread = any(k in normalized for k in ["sin leer", "no leida", "no leido", "pendiente de lectura"])
    if has_notification and has_unread:
        return ReporteResponse(
            titulo="Notificaciones sin leer",
            descripcion="Listado de notificaciones pendientes de lectura.",
            intencionDetectada="listar_notificaciones",
            entidadPrincipal="notificaciones",
            campos=["id", "usuarioId", "leida", "fechaCreacion"],
            filtros=[Filtro(campo="leida", operador="=", valor=False)],
            ordenamiento=[Ordenamiento(campo="fechaCreacion", direccion="desc")],
            limite=50,
            formatoSalida="pantalla",
            visualizacion="tabla",
            requiereAclaracion=False,
            confianza=0.95,
            motor=MotorTipo.MOTOR_FALLBACK
        )

    return None

from app.modules.reportes_dinamicos.schemas import (
    ReporteResponse, Metrica, Filtro, Ordenamiento, MotorTipo,
    PlanConsulta, AsistenteDatosResponse
)
from app.modules.reportes_dinamicos.motor_ia_client import motor_ia_client
from app.modules.reportes_dinamicos.prompt_builder import (
    build_prompt_reportes, build_prompt_asistente, build_prompt_respuesta_final
)
from app.modules.reportes_dinamicos.catalogo_reportes import validar_plan_consulta

logger = logging.getLogger(__name__)

# ============================================================
# MOTOR INTERNO (Keras) - Se carga bajo demanda
# ============================================================
_motor_interno = {
    "model": None,
    "tokenizer": None,
    "le_intent": None,
    "le_format": None,
    "loaded": False
}

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def _load_motor_interno():
    """Carga el modelo Keras interno si está disponible."""
    if _motor_interno["loaded"]:
        return _motor_interno["model"] is not None
    
    _motor_interno["loaded"] = True
    model_path = MODELS_DIR / "modelo_reportes.keras"
    tokenizer_path = MODELS_DIR / "tokenizer.pkl"
    encoders_path = MODELS_DIR / "label_encoders.pkl"
    
    if not model_path.exists():
        logger.info("Motor Interno: modelo Keras no encontrado. Se usará fallback.")
        return False
    
    try:
        from tensorflow.keras.models import load_model
        _motor_interno["model"] = load_model(str(model_path))
        
        with open(tokenizer_path, 'rb') as f:
            _motor_interno["tokenizer"] = pickle.load(f)
        
        with open(encoders_path, 'rb') as f:
            encoders = pickle.load(f)
            _motor_interno["le_intent"] = encoders['intent']
            _motor_interno["le_format"] = encoders['format']
        
        logger.info("Motor Interno: modelo Keras cargado exitosamente.")
        return True
    except Exception as e:
        logger.error(f"Motor Interno: error cargando modelo Keras: {e}")
        return False


# Mapeos de intenciones del motor interno a configuraciones de reporte
INTENT_MAPPINGS = {
    "ranking_politicas_mas_utilizadas": {
        "titulo": "Ranking de políticas más utilizadas",
        "descripcion": "Muestra las políticas con mayor cantidad de trámites iniciados.",
        "entidadPrincipal": "instancias_politica",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "cantidadTramites"}],
        "agrupaciones": ["politicaNombre"],
        "ordenamiento": [{"campo": "cantidadTramites", "direccion": "desc"}],
        "visualizacion": "grafico_barras"
    },
    "ranking_clientes_por_tramites": {
        "titulo": "Ranking de clientes por trámites",
        "descripcion": "Muestra los clientes que han iniciado más trámites.",
        "entidadPrincipal": "instancias_politica",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "cantidadTramites"}],
        "agrupaciones": ["creadaPor"],
        "ordenamiento": [{"campo": "cantidadTramites", "direccion": "desc"}],
        "visualizacion": "tabla"
    },
    "tramites_por_estado_departamento": {
        "titulo": "Trámites por estado y departamento",
        "descripcion": "Distribución de trámites según su estado y departamento.",
        "entidadPrincipal": "instancias_politica",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "cantidadTramites"}],
        "agrupaciones": ["estadoInstancia", "departamentoId"],
        "ordenamiento": [{"campo": "cantidadTramites", "direccion": "desc"}],
        "visualizacion": "grafico_pie"
    },
    "pagos_por_politica": {
        "titulo": "Pagos agrupados por política",
        "descripcion": "Total de pagos agrupados por cada política de negocio.",
        "entidadPrincipal": "pagos",
        "metricas": [{"operacion": "sum", "campo": "monto", "alias": "totalPagos"}],
        "agrupaciones": ["politicaId"],
        "ordenamiento": [{"campo": "totalPagos", "direccion": "desc"}],
        "visualizacion": "tabla"
    },
    "tareas_pendientes_funcionario": {
        "titulo": "Tareas pendientes por funcionario",
        "descripcion": "Cantidad de tareas pendientes asignadas a cada funcionario.",
        "entidadPrincipal": "tareas_actividad",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "tareasPendientes"}],
        "agrupaciones": ["responsableId"],
        "ordenamiento": [{"campo": "tareasPendientes", "direccion": "desc"}],
        "visualizacion": "tabla"
    }
}


def _extract_filters_heuristic(texto: str) -> List[dict]:
    """Extracción heurística de filtros a partir del texto (motor interno/fallback)."""
    filtros = []
    texto = texto.lower()
    
    if "este mes" in texto:
        filtros.append({"campo": "fechaCreacion", "operador": "mes_actual", "valor": None})
    elif "este año" in texto or "este ano" in texto:
        filtros.append({"campo": "fechaCreacion", "operador": "anio_actual", "valor": None})
    elif "ultimos 7 dias" in texto or "últimos 7 días" in texto:
        filtros.append({"campo": "fechaCreacion", "operador": "ultimos_dias", "valor": 7})
    elif "ultimos 30 dias" in texto or "últimos 30 días" in texto:
        filtros.append({"campo": "fechaCreacion", "operador": "ultimos_dias", "valor": 30})
    elif "ultimos 3 meses" in texto or "últimos 3 meses" in texto:
        filtros.append({"campo": "fechaCreacion", "operador": "ultimos_meses", "valor": 3})
    
    if "pendientes" in texto:
        filtros.append({"campo": "estadoInstancia", "operador": "=", "valor": "PENDIENTE"})
    elif "finalizados" in texto or "completados" in texto:
        filtros.append({"campo": "estadoInstancia", "operador": "=", "valor": "FINALIZADA"})
    elif "rechazados" in texto:
        filtros.append({"campo": "estadoInstancia", "operador": "=", "valor": "RECHAZADA"})
    
    return filtros


def _detect_format(texto: str) -> str:
    """Detecta formato de salida solicitado."""
    texto = texto.lower()
    if "excel" in texto or "xlsx" in texto:
        return "excel"
    elif "pdf" in texto:
        return "pdf"
    elif "word" in texto or "docx" in texto:
        return "word"
    return "pantalla"


# ============================================================
# INTERPRETACIÓN CON MOTOR IA AVANZADO
# ============================================================

async def interpretar_con_motor_avanzado(texto: str) -> Optional[ReporteResponse]:
    """
    Usa el Motor IA Avanzado (proveedor API) para interpretar la solicitud.
    Retorna None si no está disponible o falla.
    """
    if not motor_ia_client.is_available():
        logger.warning("Motor IA Avanzado no disponible o desactivado.")
        return None
    
    logger.info("Motor IA Avanzado: intentando interpretación...")
    
    try:
        messages = build_prompt_reportes(texto)
        resultado = await motor_ia_client.interpretar(messages)
        
        if not resultado:
            return None
        
        # Validar contra catálogo
        errores = validar_plan_consulta(resultado)
        if errores:
            logger.warning(f"Motor IA Avanzado: plan con errores de validación: {errores}")
            # Intentar corregir errores menores, fallar en errores mayores
            if any("Entidad no permitida" in e for e in errores):
                logger.error(f"Motor IA Avanzado: entidad no válida, descartando plan.")
                return None
        
        # Construir ReporteResponse desde el JSON de la IA
        response = ReporteResponse(
            titulo=resultado.get("titulo", "Reporte Dinámico"),
            descripcion=resultado.get("descripcion", ""),
            intencionDetectada=resultado.get("intencionDetectada", "interpretacion_ia"),
            entidadPrincipal=resultado.get("entidadPrincipal"),
            campos=resultado.get("campos", []),
            metricas=[Metrica(**m) for m in resultado.get("metricas", [])],
            filtros=[Filtro(**f) for f in resultado.get("filtros", [])],
            agrupaciones=resultado.get("agrupaciones", []),
            ordenamiento=[Ordenamiento(**o) for o in resultado.get("ordenamiento", [])],
            limite=min(int(resultado.get("limite") if resultado.get("limite") is not None else 50), 500),
            formatoSalida=resultado.get("formatoSalida", _detect_format(texto)),
            visualizacion=resultado.get("visualizacion", "tabla"),
            requiereAclaracion=resultado.get("requiereAclaracion", False),
            preguntaAclaratoria=resultado.get("preguntaAclaratoria"),
            opcionesSugeridas=resultado.get("opcionesSugeridas", []),
            confianza=float(resultado.get("confianza", 0.8)),
            motor=MotorTipo.MOTOR_IA_AVANZADO,
            requiereResolucionBackend=resultado.get("requiereResolucionBackend", True)
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Motor IA Avanzado: error interpretando: {e}")
        return None


# ============================================================
# INTERPRETACIÓN CON MOTOR INTERNO (Keras)
# ============================================================

def interpretar_con_motor_interno(texto: str) -> Optional[ReporteResponse]:
    """
    Usa el modelo Keras entrenado internamente para interpretar la solicitud.
    Sirve como fallback y demostración académica del motor propio.
    """
    if not _load_motor_interno():
        return None
    
    try:
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        
        model = _motor_interno["model"]
        tokenizer = _motor_interno["tokenizer"]
        le_intent = _motor_interno["le_intent"]
        le_format = _motor_interno["le_format"]
        
        texto_proc = texto.lower().strip()
        seq = tokenizer.texts_to_sequences([texto_proc])
        padded = pad_sequences(seq, maxlen=30, padding='post', truncating='post')
        
        preds = model.predict(padded, verbose=0)
        pred_intent = np.argmax(preds[0], axis=-1)[0]
        pred_format = np.argmax(preds[1], axis=-1)[0]
        pred_aclaracion = preds[2][0][0]
        
        intent_label = le_intent.inverse_transform([pred_intent])[0]
        format_label = le_format.inverse_transform([pred_format])[0]
        requiere_aclaracion = bool(pred_aclaracion > 0.5)
        confianza = float(np.max(preds[0][0]))
        
        if intent_label == "ambiguo" or requiere_aclaracion:
            # Validación de ambigüedad
            # No declarar ambiguo si hay entidad clara + condición clara
            normalized = normalize_text(texto)
            if has_entity_and_condition_clear(normalized):
                logger.warning(f"Motor Interno: Se detectó ambiguo con alta confianza ({confianza:.4f}), pero la frase '{texto}' parece específica. Ignorando respuesta del motor interno y aplicando reglas semánticas.")
                res_rules = aplicar_reglas_semanticas_basicas(texto)
                if res_rules:
                    return res_rules
                return None
            
            return ReporteResponse(
                requiereAclaracion=True,
                preguntaAclaratoria="¿Podrías dar más detalles sobre lo que deseas analizar? (Ej. agrupado por cliente, política, período, etc.)",
                confianza=confianza,
                motor=MotorTipo.MOTOR_INTERNO
            )
        
        mapping = INTENT_MAPPINGS.get(intent_label, {})
        if not mapping:
            return ReporteResponse(
                requiereAclaracion=True,
                preguntaAclaratoria="No se pudo determinar la intención con el motor interno. Intenta ser más específico.",
                confianza=confianza,
                motor=MotorTipo.MOTOR_INTERNO
            )
        
        filtros = _extract_filters_heuristic(texto)
        
        return ReporteResponse(
            titulo=mapping.get("titulo", "Reporte Dinámico"),
            descripcion=mapping.get("descripcion", f"Interpretación de: '{texto}'"),
            intencionDetectada=intent_label,
            entidadPrincipal=mapping.get("entidadPrincipal"),
            metricas=[Metrica(**m) for m in mapping.get("metricas", [])],
            agrupaciones=mapping.get("agrupaciones", []),
            ordenamiento=[Ordenamiento(**o) for o in mapping.get("ordenamiento", [])],
            filtros=[Filtro(**f) for f in filtros],
            formatoSalida=format_label if format_label != "ambiguo" else _detect_format(texto),
            visualizacion=mapping.get("visualizacion", "tabla"),
            confianza=confianza,
            requiereAclaracion=False,
            motor=MotorTipo.MOTOR_INTERNO
        )
        
    except Exception as e:
        logger.error(f"Motor Interno: error interpretando: {e}")
        return None


# ============================================================
# MOTOR FALLBACK
# ============================================================

def interpretar_con_fallback(texto: str) -> ReporteResponse:
    """
    Respuesta controlada cuando ningún motor está disponible.
    Nunca rompe la aplicación.
    """
    formato = _detect_format(texto)
    filtros = _extract_filters_heuristic(texto)
    
    return ReporteResponse(
        titulo="Reporte Solicitado",
        descripcion=f"Se recibió la consulta: '{texto}'. Se utilizó procesamiento básico.",
        intencionDetectada="fallback",
        requiereAclaracion=True,
        preguntaAclaratoria=(
            "El motor de interpretación avanzada no está disponible en este momento. "
            "Por favor, intenta con consultas más específicas como: "
            "'Mostrar las políticas más usadas este mes' o "
            "'Listar trámites pendientes agrupados por departamento'."
        ),
        opcionesSugeridas=[
            "Políticas más usadas este mes",
            "Trámites pendientes por departamento",
            "Pagos agrupados por política",
            "Tareas pendientes por funcionario",
            "Clientes con más trámites este año"
        ],
        formatoSalida=formato,
        filtros=[Filtro(**f) for f in filtros],
        confianza=0.0,
        motor=MotorTipo.MOTOR_FALLBACK
    )


# ============================================================
# ASISTENTE DE DATOS - PLANIFICACIÓN
# ============================================================

async def planificar_consulta_asistente(texto: str, contexto: str = None) -> Optional[PlanConsulta]:
    """
    Genera un plan de consulta para el asistente de datos usando el Motor IA Avanzado.
    """
    if not motor_ia_client.is_available():
        return None
    
    try:
        messages = build_prompt_asistente(texto, contexto)
        resultado = await motor_ia_client.interpretar(messages)
        
        if not resultado:
            return None
        
        # Validar contra catálogo
        errores = validar_plan_consulta(resultado)
        if errores:
            logger.warning(f"Asistente Datos: plan con errores: {errores}")
        
        plan = PlanConsulta(
            requiereDatos=resultado.get("requiereDatos", True),
            tipoConsulta=resultado.get("tipoConsulta", "analitica"),
            fuentesNecesarias=resultado.get("fuentesNecesarias", []),
            entidadPrincipal=resultado.get("entidadPrincipal", ""),
            operacion=resultado.get("operacion", "listado"),
            camposSolicitados=resultado.get("camposSolicitados", []),
            filtros=[Filtro(**f) for f in resultado.get("filtros", [])],
            agrupaciones=resultado.get("agrupaciones", []),
            ordenamiento=[Ordenamiento(**o) for o in resultado.get("ordenamiento", [])],
            limite=min(int(resultado.get("limite") if resultado.get("limite") is not None else 50), 500),
            requiereBusquedaSemantica=resultado.get("requiereBusquedaSemantica", False),
            requiereAclaracion=resultado.get("requiereAclaracion", False),
            preguntaAclaratoria=resultado.get("preguntaAclaratoria")
        )
        
        return plan
        
    except Exception as e:
        logger.error(f"Asistente Datos: error planificando: {e}")
        return None


async def generar_respuesta_natural(texto_original: str, datos: list) -> Dict[str, Any]:
    """
    Genera una respuesta en lenguaje natural basada en los datos recuperados.
    Usa el Motor IA Avanzado si está disponible, sino genera una respuesta básica.
    """
    if motor_ia_client.is_available() and datos:
        try:
            messages = build_prompt_respuesta_final(texto_original, datos)
            resultado = await motor_ia_client.interpretar(messages)
            if resultado:
                return resultado
        except Exception as e:
            logger.error(f"Error generando respuesta natural: {e}")
    
    # Fallback: respuesta básica
    if not datos:
        return {
            "respuesta": "No se encontraron resultados para los criterios indicados.",
            "resumen": "Sin resultados.",
            "visualizacionSugerida": "tabla",
            "accionesSugeridas": [],
            "advertencias": []
        }
    
    return {
        "respuesta": f"Se encontraron {len(datos)} registros para tu consulta.",
        "resumen": f"{len(datos)} resultados encontrados.",
        "visualizacionSugerida": "tabla",
        "accionesSugeridas": ["Exportar a Excel", "Ver detalle"],
        "advertencias": []
    }


# ============================================================
# ORQUESTADOR PRINCIPAL
# ============================================================

async def interpretar_reporte(texto: str) -> ReporteResponse:
    logger.info("=========================================")
    logger.info("== INICIANDO ORQUESTACIÓN DE REPORTE ==")
    logger.info(f"Texto recibido: '{texto}'")
    
    # Obtener info de catálogo y configuración
    from app.modules.reportes_dinamicos.catalogo_reportes import get_catalogo_completo
    from app.modules.reportes_dinamicos.config import reportes_settings
    
    catalogo = get_catalogo_completo()
    cat_origen = catalogo.get("origen", "FALLBACK_LOCAL")
    logger.info(f"Catálogo de origen usado: {cat_origen}")
    
    api_key_presente = bool(motor_ia_client.api_key and len(motor_ia_client.api_key) > 5)
    motor_avanzado_habilitado = reportes_settings.ia_enabled
    logger.info(f"Motor avanzado habilitado: {motor_avanzado_habilitado}")
    logger.info(f"API key presente: {api_key_presente}")
    logger.info(f"Base URL configurada: {motor_ia_client.base_url}")
    logger.info(f"Modelo configurado: {motor_ia_client.model}")
    
    # 1. Motor IA Avanzado
    logger.info("Intento de llamada a Motor IA Avanzado...")
    resultado_avanzado = None
    motivo_fallo_avanzado = None
    
    if not motor_avanzado_habilitado:
        motivo_fallo_avanzado = "Motor avanzado deshabilitado en configuración (DEEPSEEK_ENABLED=false)"
    elif not api_key_presente:
        motivo_fallo_avanzado = "Falta la API key o es demasiado corta"
    else:
        try:
            resultado_avanzado = await interpretar_con_motor_avanzado(texto)
            if resultado_avanzado is None:
                motivo_fallo_avanzado = "La llamada retornó None (posible error de HTTP, timeout o formato JSON no válido)"
        except Exception as e:
            motivo_fallo_avanzado = f"Excepción durante interpretación avanzada: {str(e)}"
            
    if resultado_avanzado is not None:
        logger.info("Resultado del motor avanzado: Plan de consulta interpretado correctamente.")
        logger.info(f"Resultado final -> motor: {resultado_avanzado.motor}, intención: {resultado_avanzado.intencionDetectada}, confianza: {resultado_avanzado.confianza:.2f}")
        logger.info("=========================================")
        return resultado_avanzado
        
    logger.warning(f"MOTOR_IA_AVANZADO falló. Motivo exacto: {motivo_fallo_avanzado}")
    
    # 2. Reglas semánticas básicas (Fallback Semántico)
    logger.info("Evaluando reglas semánticas básicas...")
    resultado_semantico = aplicar_reglas_semanticas_basicas(texto)
    if resultado_semantico is not None:
        logger.info("Interpretación exitosa mediante REGLAS SEMÁNTICAS (Fallback)")
        logger.info(f"Resultado final -> motor: {resultado_semantico.motor}, intención: {resultado_semantico.intencionDetectada}, confianza: {resultado_semantico.confianza:.2f}")
        logger.info("=========================================")
        return resultado_semantico
        
    # 3. Motor Interno (Keras)
    logger.info("Reglas semánticas no aplicaron. Intentando Motor Interno (Keras)...")
    
    resultado_interno = None
    motivo_fallo_interno = None
    
    # Verificar si el motor interno está disponible
    from pathlib import Path
    model_path = Path(__file__).resolve().parent / "models" / "modelo_reportes.keras"
    if not model_path.exists():
        motivo_fallo_interno = "Archivo de modelo Keras no encontrado"
    else:
        try:
            resultado_interno = interpretar_con_motor_interno(texto)
            if resultado_interno is None:
                motivo_fallo_interno = "Interpretación anulada (ej. clasificada como ambiguo pero con entidad + condición clara)"
        except Exception as e:
            motivo_fallo_interno = f"Excepción en motor interno Keras: {str(e)}"
            
    if resultado_interno is not None:
        logger.info("Resultado del motor interno: Plan de consulta interpretado.")
        logger.info(f"Resultado final -> motor: {resultado_interno.motor}, intención: {resultado_interno.intencionDetectada}, confianza: {resultado_interno.confianza:.2f}")
        logger.info("=========================================")
        return resultado_interno
        
    logger.warning(f"Cae a MOTOR_INTERNO falló. Motivo exacto: {motivo_fallo_interno}")
    
    # 4. Fallback General
    logger.warning("Cae a MOTOR_FALLBACK por defecto. Motivo: Ningún motor pudo procesar la consulta.")
    resultado_fallback = interpretar_con_fallback(texto)
    logger.info(f"Resultado final -> motor: {resultado_fallback.motor}, intención: {resultado_fallback.intencionDetectada}, confianza: {resultado_fallback.confianza:.2f}")
    logger.info("=========================================")
    return resultado_fallback


async def procesar_pregunta_asistente(texto: str, contexto: str = None) -> dict:
    """
    Procesa una pregunta libre del asistente de datos.
    Retorna el plan de consulta para que el backend lo ejecute.
    """
    plan = await planificar_consulta_asistente(texto, contexto)
    
    if plan is None:
        # Sin Motor IA Avanzado, retornar plan básico
        return {
            "plan": PlanConsulta(
                requiereAclaracion=True,
                preguntaAclaratoria=(
                    "El motor de interpretación avanzada no está disponible. "
                    "Intenta con preguntas más específicas."
                )
            ).model_dump(),
            "motor": MotorTipo.MOTOR_FALLBACK
        }
    
    return {
        "plan": plan.model_dump(),
        "motor": MotorTipo.MOTOR_IA_AVANZADO
    }
