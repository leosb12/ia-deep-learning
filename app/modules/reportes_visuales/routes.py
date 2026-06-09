from fastapi import APIRouter, HTTPException
import logging
import re
from typing import List, Dict, Any
from app.modules.reportes_visuales.schemas import ReporteVisualRequest, ReporteVisualResponse, BloqueReporteIntent
from app.modules.reportes_visuales.predictor_reporte_visual import predictor_visual
from app.modules.reportes_dinamicos.motor_ia_client import motor_ia_client
from app.modules.reportes_dinamicos.catalogo_reportes import get_catalogo_completo

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ia/reportes-visuales", tags=["Reportes Visuales Inteligentes"])

METRICAS_DISPONIBLES = [
    "funcionarios_mas_activos", "clientes_mas_inician_politicas", "administradores_mas_politicas_crearon",
    "politicas_mas_usadas", "politicas_por_estado", "tramites_por_estado", "tramites_por_mes",
    "tramites_por_departamento", "tramites_por_prioridad", "tramites_finalizados_por_funcionario",
    "promedio_tiempo_finalizacion", "pagos_por_estado", "pagos_por_politica", "total_tramites",
    "total_politicas", "total_usuarios", "total_pagos"
]

@router.post("/interpretar", response_model=ReporteVisualResponse)
async def interpretar_reporte_visual(req: ReporteVisualRequest):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")

    logger.info(f"== INTERPRETAR REPORTE VISUAL == Prompt: '{req.prompt}', iaPlus: {req.iaPlus}")

    # 1. Intentar con DeepSeek (Motor IA Avanzado)
    if motor_ia_client.is_available():
        try:
            catalogo = get_catalogo_completo()
            if req.iaPlus:
                system_prompt = f"""Eres un generador de reportes visuales dinámicos simulados (modo IA+).
Tu tarea es leer la solicitud del administrador y convertirla en un JSON estructurado de bloques de reporte, incluyendo datos simulados realistas para cada bloque de acuerdo a lo que el usuario pida.

No te limites a las métricas reales de la base de datos. Puedes inventar nombres de métricas, intenciones y campos para adaptarte exactamente a lo que pide el usuario (por ejemplo, "requisitoInicial", "porcentajeObligatoriedad", "frecuenciaUso", etc.).

Si el usuario solicita información sobre políticas de negocio o usuarios (clientes/funcionarios/administradores), debes mapear e incluir únicamente los nombres reales que se te proporcionan en los catálogos. No inventes nombres de políticas ni de usuarios. Cualquier otra métrica, valor numérico, estado o etiqueta no identificatoria debe ser inventado y simulado de forma realista.

Debes devolver este formato exacto:
{{
  "titulo": "Título descriptivo del reporte",
  "descripcion": "Descripción del reporte según la solicitud",
  "bloques": [
    {{
      "tipo": "bar | pie | doughnut | line | area | table | matrix | kpi",
      "intencion": "nombre_metrica",
      "titulo": "Título del bloque",
      "orden": 1,
      "entidadPrincipal": "nombre_coleccion",
      "metrica": "nombre_metrica",
      "limite": 10,
      "filtros": {{}},
      "datos": {{
        "labels": ["etiqueta1", "etiqueta2", ...],
        "values": [45, 30, ...],
        "columns": ["columna1", "columna2", ...],
        "rows": [
          ["valor1", "valor2", ...],
          ...
        ]
      }}
    }}
  ]
}}

Catálogos de datos reales disponibles para usar como categorías o campos identificatorios:
- Políticas reales: {req.politicasReales or []}
- Usuarios reales: {req.usuariosReales or []}

Reglas:
- Los tipos de bloque permitidos son: bar, pie, doughnut, line, area, table, matrix, kpi.
- En el campo "datos":
  - Para kpi: "labels" debe contener el nombre de la métrica (ej. ["Total"]), "values" debe contener un único valor numérico (ej. [150]). "columns" y "rows" vacíos.
  - Para gráficos (bar, pie, doughnut, line, area): "labels" debe contener las categorías (e.g. nombres reales de políticas o nombres reales de usuarios), "values" debe contener los valores correspondientes. "columns" y "rows" vacíos.
  - Para table o matrix: "columns" debe tener la lista de nombres de columnas. "rows" debe tener la lista de filas, donde cada fila es una lista de valores en el mismo orden que las columnas. "labels" y "values" vacíos.
- Genera datos simulados lógicos y coherentes que representen fielmente la solicitud.
- Devuelve únicamente JSON válido. Sin markdown, sin explicaciones.
"""
            else:
                system_prompt = f"""Eres un generador de estructuras JSON para reportes visuales dinámicos.

Tu tarea es leer la solicitud del administrador y convertirla en un JSON estructurado de bloques de reporte.

No generes consultas MongoDB.
No inventes colecciones.
No inventes campos.
No inventes datos.
No devuelvas texto explicativo.
Devuelve únicamente JSON válido.

Tipos de bloque permitidos:
bar, pie, doughnut, line, area, table, matrix, kpi.

Catálogo de datos disponible:
{catalogo}

Métricas disponibles:
{METRICAS_DISPONIBLES}

Debes devolver este formato exacto:
{{
  "titulo": "Título descriptivo del reporte",
  "descripcion": "Descripción del reporte según la solicitud",
  "bloques": [
    {{
      "tipo": "bar | pie | doughnut | line | area | table | matrix | kpi",
      "intencion": "nombre_metrica",
      "titulo": "Título del bloque",
      "orden": 1,
      "entidadPrincipal": "nombre_coleccion",
      "metrica": "nombre_metrica",
      "limite": 10,
      "filtros": {{}}
    }}
  ]
}}

Reglas:
- Respeta el orden indicado por el usuario.
- Si dice arriba, primero o al inicio, pon orden menor.
- Si dice abajo, debajo, al final, pon orden mayor.
- Si pide torta, usa pie.
- Si pide dona, usa doughnut.
- Si pide barras, usa bar.
- Si pide tabla o matriz, usa table o matrix.
- Si pide indicador, total, cantidad general o KPI, usa kpi.
- Usa solamente métricas disponibles.
- Si el usuario pide algún campo o métrica que no está en el catálogo, agrega un bloque con tipo "error" y explica el problema en la intención.
"""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": req.prompt}
            ]
            resultado = await motor_ia_client.interpretar(messages)
            if resultado and "bloques" in resultado:
                # Validar campos mínimos
                titulo = resultado.get("titulo", "Reporte inteligente personalizado")
                descripcion = resultado.get("descripcion", "Reporte generado según instrucción")
                bloques_list = []
                for b in resultado["bloques"]:
                    # Validar tipo de bloque
                    tipo = b.get("tipo", "table")
                    if tipo not in ["bar", "pie", "doughnut", "line", "area", "table", "matrix", "kpi", "error"]:
                        tipo = "table"
                    
                    bloques_list.append(BloqueReporteIntent(
                        tipo=tipo,
                        intencion=b.get("intencion", b.get("metrica", "desconocida")),
                        titulo=b.get("titulo", "Bloque de Reporte"),
                        orden=int(b.get("orden", 1)),
                        entidadPrincipal=b.get("entidadPrincipal", "instancias_politica"),
                        metrica=b.get("metrica", b.get("intencion", "desconocida")),
                        limite=int(b.get("limite", 10)),
                        filtros=b.get("filtros"),
                        datos=b.get("datos")
                    ))
                return ReporteVisualResponse(
                    titulo=titulo,
                    descripcion=descripcion,
                    bloques=bloques_list
                )
        except Exception as e:
            logger.error(f"Error en interpretación avanzada de reporte visual: {e}")

    # 2. Fallback Heurístico con el Modelo Keras Entrenado
    logger.info("Usando Fallback Heurístico + Modelo Keras para reporte visual...")
    # Dividir el prompt por frases comunes de separación
    partes = re.split(r'\b(?:abajo|debajo|y al final|despues|luego|y al lado|y debajo|antes de|y por ultimo|,)\b', req.prompt.lower())
    partes = [p.strip() for p in partes if len(p.strip()) > 5]

    if not partes:
        partes = [req.prompt]

    bloques_detectados = []
    orden = 1
    
    # Asegurar que el predictor está cargado
    predictor_visual.load()

    for parte in partes:
        # Usar el predictor Keras para clasificar tipo de gráfico, intención, entidad y métrica
        pred = predictor_visual.predecir_bloque(parte)
        if pred:
            # Crear títulos legibles basados en la métrica
            titulo_bloque = pred["intencion"].replace("_", " ").capitalize()
            bloques_detectados.append(BloqueReporteIntent(
                tipo=pred["tipo"],
                intencion=pred["intencion"],
                titulo=titulo_bloque,
                orden=orden,
                entidadPrincipal=pred["entidadPrincipal"],
                metrica=pred["metrica"],
                limite=10,
                filtros={}
            ))
            orden += 1

    # Si no se detectó nada, poner un bloque por defecto
    if not bloques_detectados:
        bloques_detectados.append(BloqueReporteIntent(
            tipo="table",
            intencion="tramites_por_estado",
            titulo="Trámites por Estado",
            orden=1,
            entidadPrincipal="instancias_politica",
            metrica="tramites_por_estado",
            limite=10,
            filtros={}
        ))

    return ReporteVisualResponse(
        titulo="Reporte Inteligente (Fallback)",
        descripcion="Interpretado mediante modelo local Keras y heurísticas.",
        bloques=bloques_detectados
    )
