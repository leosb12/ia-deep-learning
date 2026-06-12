from fastapi import APIRouter, HTTPException, Request
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
    "total_politicas", "total_usuarios", "total_pagos", "cuellos_botella"
]

@router.post("/interpretar", response_model=ReporteVisualResponse)
async def interpretar_reporte_visual(req: ReporteVisualRequest, request: Request):
    if not req.prompt or not req.prompt.strip():
        raise HTTPException(status_code=400, detail="El prompt no puede estar vacío.")

    # Detectar headers de forzado offline para saltar a DeepSeek/Cloud
    offline_mode = (
        request.headers.get("X-Offline-Mode") == "true" or
        request.headers.get("X-Local-Deep-Learning-Only") == "true" or
        request.headers.get("X-Disable-Cloud-AI") == "true"
    )

    logger.info(f"== INTERPRETAR REPORTE VISUAL == Prompt: '{req.prompt}', iaPlus: {req.iaPlus}, offline_mode: {offline_mode}")

    # Normalizador semántico local offline para las 20 peticiones clásicas
    if offline_mode:
        def normalizar_texto(texto: str) -> str:
            texto = texto.lower()
            replacements = {
                "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
                "ü": "u", "ñ": "n"
            }
            for char, replacement in replacements.items():
                texto = texto.replace(char, replacement)
            return texto

        prompt_clean = normalizar_texto(req.prompt).strip()
        
        # Bloquear reportes de pagos en modo offline
        if any(w in prompt_clean for w in ["pago", "pagos", "cobro", "cobros", "transaccion", "transacciones", "stripe", "monto", "montos", "recaudacion", "recaud"]):
            return ReporteVisualResponse(
                titulo="Reporte Incompatible",
                descripcion="El modo offline no permite acceder a información financiera o de pagos.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="error",
                        intencion="error",
                        titulo="Métrica no disponible",
                        orden=1,
                        entidadPrincipal="pagos",
                        metrica="total_pagos",
                        limite=1,
                        filtros={}
                    )
                ]
            )

        # 1. Muéstrame trámites por estado
        if any(p in prompt_clean for p in ["tramites por estado", "tramite por estado", "solicitudes por estado", "solicitud por estado"]):
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="doughnut" if "dona" in prompt_clean or "torta" in prompt_clean else "bar",
                        intencion="tramites_por_estado",
                        titulo="Trámites por Estado",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_por_estado",
                        limite=10,
                        filtros={}
                    )
                ]
            )

        # 2. Quiero trámites por mes
        if any(p in prompt_clean for p in ["tramites por mes", "tramite por mes", "solicitudes por mes", "solicitud por mes"]):
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="line",
                        intencion="tramites_por_mes",
                        titulo="Trámites por Mes",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_por_mes",
                        limite=12,
                        filtros={}
                    )
                ]
            )

        # 3. Genera un KPI con el total de trámites
        if "total de tramites" in prompt_clean or "kpi con el total de tramites" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="kpi",
                        intencion="total_tramites",
                        titulo="Total de Trámites",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="total_tramites",
                        limite=1,
                        filtros={}
                    )
                ]
            )

        # 5. Trámites finalizados por funcionario
        if "finalizados por funcionario" in prompt_clean or "completados por funcionario" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="table" if "tabla" in prompt_clean or "matriz" in prompt_clean else "bar",
                        intencion="tramites_finalizados_por_funcionario",
                        titulo="Trámites Finalizados por Funcionario",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_finalizados_por_funcionario",
                        limite=10,
                        filtros={"estadoInstancia": "FINALIZADA"}
                    )
                ]
            )

        # 6. Trámites pendientes por funcionario
        if "pendientes por funcionario" in prompt_clean or "en curso por funcionario" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="table" if "tabla" in prompt_clean or "matriz" in prompt_clean else "bar",
                        intencion="funcionarios_mas_activos",
                        titulo="Trámites Pendientes por Funcionario",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="funcionarios_mas_activos",
                        limite=10,
                        filtros={"estadoInstancia": "PENDIENTE"}
                    )
                ]
            )

        # 7. Trámites rechazados por funcionario
        if "rechazados por funcionario" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="table" if "tabla" in prompt_clean or "matriz" in prompt_clean else "bar",
                        intencion="funcionarios_mas_activos",
                        titulo="Trámites Rechazados por Funcionario",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="funcionarios_mas_activos",
                        limite=10,
                        filtros={"estadoInstancia": "RECHAZADO"}
                    )
                ]
            )

        # 17. Ranking de funcionarios por trámites completados
        if "ranking de funcionarios por tramites completados" in prompt_clean or "ranking de funcionarios por tramites finalizados" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="bar",
                        intencion="tramites_finalizados_por_funcionario",
                        titulo="Ranking de Funcionarios por Trámites Completados",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_finalizados_por_funcionario",
                        limite=10,
                        filtros={}
                    )
                ]
            )

        # 4. Quiero funcionarios más activos
        if "funcionarios mas activos" in prompt_clean or "funcionario mas activo" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="bar",
                        intencion="funcionarios_mas_activos",
                        titulo="Funcionarios más Activos",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="funcionarios_mas_activos",
                        limite=10,
                        filtros={}
                    )
                ]
            )

        # Caso especial para "tramites finalizados este mes" (independiente de funcionario)
        if "finalizados este mes" in prompt_clean and "funcionario" not in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="kpi",
                        intencion="total_tramites",
                        titulo="Trámites Finalizados este Mes",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="total_tramites",
                        limite=1,
                        filtros={"estadoInstancia": "FINALIZADA", "mesActual": True}
                    )
                ]
            )

        # Caso especial para "ultimos meses"
        if "ultimos meses" in prompt_clean or "ultimo mes" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="line",
                        intencion="tramites_por_mes",
                        titulo="Evolución en los Últimos Meses",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_por_mes",
                        limite=12,
                        filtros={}
                    )
                ]
            )

        # 16. Trámites finalizados este mes por funcionario
        if "finalizados este mes por funcionario" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="table" if "tabla" in prompt_clean else "bar",
                        intencion="tramites_finalizados_por_funcionario",
                        titulo="Trámites Finalizados este Mes por Funcionario",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_finalizados_por_funcionario",
                        limite=10,
                        filtros={"estadoInstancia": "FINALIZADA", "mesActual": True}
                    )
                ]
            )

        # 15. Trámites durante este mes por estado
        if "durante este mes por estado" in prompt_clean or "este mes por estado" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="doughnut" if "dona" in prompt_clean or "torta" in prompt_clean else "bar",
                        intencion="tramites_por_estado",
                        titulo="Trámites del Mes por Estado",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_por_estado",
                        limite=10,
                        filtros={"mesActual": True}
                    )
                ]
            )

        # 18. Distribución de trámites por estado durante este mes
        if "distribucion de tramites por estado" in prompt_clean and ("este mes" in prompt_clean or "mes actual" in prompt_clean):
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="doughnut" if "dona" in prompt_clean or "torta" in prompt_clean else "pie",
                        intencion="tramites_por_estado",
                        titulo="Distribución de Trámites por Estado este Mes",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_por_estado",
                        limite=10,
                        filtros={"mesActual": True}
                    )
                ]
            )

        # 9. Políticas más utilizadas
        if "politicas mas utilizadas" in prompt_clean or "politicas mas usadas" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="bar",
                        intencion="politicas_mas_usadas",
                        titulo="Políticas más Utilizadas",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="politicas_mas_usadas",
                        limite=10,
                        filtros={}
                    )
                ]
            )

        # 19. Tabla de políticas con más trámites iniciados
        if "tabla de politicas con mas tramites" in prompt_clean or "tabla de politicas" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="table",
                        intencion="politicas_mas_usadas",
                        titulo="Políticas con más Trámites Iniciados",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="politicas_mas_usadas",
                        limite=10,
                        filtros={}
                    )
                ]
            )

        # 8. Cantidad de trámites por política
        if "tramites por politica" in prompt_clean or "tramite por politica" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="bar" if "barra" in prompt_clean else "table",
                        intencion="politicas_mas_usadas",
                        titulo="Cantidad de Trámites por Política",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="politicas_mas_usadas",
                        limite=10,
                        filtros={}
                    )
                ]
            )

        # 10. Cantidad de trámites por departamento
        if "tramites por departamento" in prompt_clean or "tramite por departamento" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="bar",
                        intencion="tramites_por_departamento",
                        titulo="Cantidad de Trámites por Departamento",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_por_departamento",
                        limite=10,
                        filtros={}
                    )
                ]
            )

        # 11. Usuarios que más inician políticas
        if "usuarios que mas inician" in prompt_clean or "usuarios iniciadores" in prompt_clean or "usuarios que mas inician politicas" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="doughnut" if "dona" in prompt_clean or "torta" in prompt_clean else "pie",
                        intencion="clientes_mas_inician_politicas",
                        titulo="Usuarios que más Inician Políticas",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="clientes_mas_inician_politicas",
                        limite=10,
                        filtros={}
                    )
                ]
            )

        # 12. Matriz de funcionarios y cantidad de trámites finalizados
        if "matriz de funcionarios" in prompt_clean or "matriz de funcionarios y cantidad" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="table",
                        intencion="tramites_finalizados_por_funcionario",
                        titulo="Matriz de Funcionarios y Trámites Finalizados",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_finalizados_por_funcionario",
                        limite=10,
                        filtros={}
                    )
                ]
            )

        # 13. Dashboard de funcionarios más activos y políticas más usadas
        if "dashboard de funcionarios mas activos y politicas" in prompt_clean or "funcionarios mas activos y politicas mas usadas" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="bar",
                        intencion="funcionarios_mas_activos",
                        titulo="Funcionarios más Activos",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="funcionarios_mas_activos",
                        limite=5,
                        filtros={}
                    ),
                    BloqueReporteIntent(
                        tipo="table",
                        intencion="politicas_mas_usadas",
                        titulo="Políticas más Usadas",
                        orden=2,
                        entidadPrincipal="instancias_politica",
                        metrica="politicas_mas_usadas",
                        limite=5,
                        filtros={}
                    )
                ]
            )

        # 14. Dashboard con KPI total de trámites, línea por mes y dona por estado
        if "dashboard con kpi total de tramites" in prompt_clean or ("kpi total de tramites" in prompt_clean and "linea por mes" in prompt_clean):
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="kpi",
                        intencion="total_tramites",
                        titulo="Total de Trámites",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="total_tramites",
                        limite=1,
                        filtros={}
                    ),
                    BloqueReporteIntent(
                        tipo="line",
                        intencion="tramites_por_mes",
                        titulo="Trámites por Mes",
                        orden=2,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_por_mes",
                        limite=12,
                        filtros={}
                    ),
                    BloqueReporteIntent(
                        tipo="doughnut",
                        intencion="tramites_por_estado",
                        titulo="Trámites por Estado",
                        orden=3,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_por_estado",
                        limite=10,
                        filtros={}
                    )
                ]
            )

        # 20. Resumen general de trámites
        if "resumen general de tramites" in prompt_clean or "resumen general" in prompt_clean:
            return ReporteVisualResponse(
                titulo="Reporte Inteligente",
                descripcion="Interpretado mediante normalizador local.",
                bloques=[
                    BloqueReporteIntent(
                        tipo="kpi",
                        intencion="total_tramites",
                        titulo="Total de Trámites",
                        orden=1,
                        entidadPrincipal="instancias_politica",
                        metrica="total_tramites",
                        limite=1,
                        filtros={}
                    ),
                    BloqueReporteIntent(
                        tipo="kpi",
                        intencion="total_tramites",
                        titulo="Finalizados",
                        orden=2,
                        entidadPrincipal="instancias_politica",
                        metrica="total_tramites",
                        limite=1,
                        filtros={"estadoInstancia": "FINALIZADA"}
                    ),
                    BloqueReporteIntent(
                        tipo="kpi",
                        intencion="total_tramites",
                        titulo="Pendientes",
                        orden=3,
                        entidadPrincipal="instancias_politica",
                        metrica="total_tramites",
                        limite=1,
                        filtros={"estadoInstancia": "EN_CURSO"}
                    ),
                    BloqueReporteIntent(
                        tipo="doughnut",
                        intencion="tramites_por_estado",
                        titulo="Distribución por Estado",
                        orden=4,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_por_estado",
                        limite=10,
                        filtros={}
                    ),
                    BloqueReporteIntent(
                        tipo="line",
                        intencion="tramites_por_mes",
                        titulo="Evolución Mensual",
                        orden=5,
                        entidadPrincipal="instancias_politica",
                        metrica="tramites_por_mes",
                        limite=12,
                        filtros={}
                    )
                ]
            )

    # 1. Intentar con DeepSeek (Motor IA Avanzado)
    if not offline_mode and motor_ia_client.is_available():
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
- REGLAS DE FECHAS (MUY IMPORTANTE):
  - Cualquier fecha de inicio, creación, finalización o registro en los datos simulados DEBE estar estrictamente en el rango del 15 de abril de 2026 al 11 de junio de 2026. Ninguna fecha puede ser anterior al 15 de abril de 2026 ni posterior al 11 de junio de 2026.
- REGLAS DE CANTIDADES (MUY IMPORTANTE):
  - Para cualquier dato de cantidad, total de trámites iniciados, tareas realizadas o similar, NUNCA uses miles. Genera valores numéricos pequeños y realistas con un tope máximo de 200 en total.
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
    
    def corregir_prediccion_heuristica(prompt_part: str, pred: dict) -> dict:
        def normalizar_texto(texto: str) -> str:
            texto = texto.lower()
            replacements = {
                "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
                "ü": "u", "ñ": "n"
            }
            for char, replacement in replacements.items():
                texto = texto.replace(char, replacement)
            return texto

        prompt_lower = normalizar_texto(prompt_part).strip()
        
        # 1. Caso especial: Cuellos de botella
        if any(k in prompt_lower for k in ["cuello de botella", "cuellos de botella", "bottleneck"]):
            return {
                "tipo": "table",
                "intencion": "cuellos_botella",
                "entidadPrincipal": "instancias_politica",
                "metrica": "cuellos_botella",
                "titulo": "Trámites con cuello de botella",
                "filtros": {}
            }
            
        # 2. Caso especial: Funcionarios + finalizados/completados/terminados/concluidos
        es_funcionario = "funcionario" in prompt_lower or "funcionarios" in prompt_lower
        es_finalizado = any(k in prompt_lower for k in ["finalizado", "terminado", "completado", "concluido", "finalizados", "terminados", "completados", "concluidos"])
        
        if es_funcionario and es_finalizado:
            tipo = "table"
            if any(k in prompt_lower for k in ["barra", "barras", "bar"]):
                tipo = "bar"
            elif any(k in prompt_lower for k in ["torta", "dona", "pie", "doughnut"]):
                tipo = "pie"
            elif "ranking" in prompt_lower:
                tipo = "bar"
                
            return {
                "tipo": tipo,
                "intencion": "tramites_finalizados_por_funcionario",
                "entidadPrincipal": "instancias_politica",
                "metrica": "tramites_finalizados_por_funcionario",
                "titulo": "Trámites finalizados por funcionario",
                "filtros": {"estadoInstancia": "FINALIZADA"}
            }

        # 3. Caso especial: Funcionarios sin "finalizado" (ej: funcionarios más activos o ranking)
        if es_funcionario:
            tipo = "bar"
            if any(k in prompt_lower for k in ["matriz", "tabla", "table"]):
                tipo = "table"
            elif any(k in prompt_lower for k in ["torta", "dona", "pie", "doughnut"]):
                tipo = "pie"
                
            return {
                "tipo": tipo,
                "intencion": "funcionarios_mas_activos",
                "entidadPrincipal": "instancias_politica",
                "metrica": "funcionarios_mas_activos",
                "titulo": "Funcionarios más activos",
                "filtros": {}
            }

        # 4. Caso especial: Trámites por estado
        if "tramite" in prompt_lower or "tramites" in prompt_lower or "solicitud" in prompt_lower or "solicitudes" in prompt_lower:
            if "estado" in prompt_lower or "estados" in prompt_lower:
                tipo = "pie"
                if any(k in prompt_lower for k in ["barra", "barras", "bar"]):
                    tipo = "bar"
                elif any(k in prompt_lower for k in ["matriz", "tabla", "table"]):
                    tipo = "table"
                    
                return {
                    "tipo": tipo,
                    "intencion": "tramites_por_estado",
                    "entidadPrincipal": "instancias_politica",
                    "metrica": "tramites_por_estado",
                    "titulo": "Trámites por estado",
                    "filtros": {}
                }

        # 5. Caso especial: total de trámites/KPIs
        if "total" in prompt_lower or "cantidad" in prompt_lower or "cuantos" in prompt_lower or "número" in prompt_lower or "numero" in prompt_lower:
            if "tramite" in prompt_lower or "tramites" in prompt_lower or "solicitud" in prompt_lower or "solicitudes" in prompt_lower:
                if not es_funcionario:
                    return {
                        "tipo": "kpi",
                        "intencion": "total_tramites",
                        "entidadPrincipal": "instancias_politica",
                        "metrica": "total_tramites",
                        "titulo": "Total de trámites",
                        "filtros": {}
                    }

        # 6. Caso especial: trámites por mes / mensual
        if "mes" in prompt_lower or "mensual" in prompt_lower or "meses" in prompt_lower:
            if "tramite" in prompt_lower or "tramites" in prompt_lower:
                return {
                    "tipo": "line" if "linea" in prompt_lower or "grafico de linea" in prompt_lower else "bar",
                    "intencion": "tramites_por_mes",
                    "entidadPrincipal": "instancias_politica",
                    "metrica": "tramites_por_mes",
                    "titulo": "Trámites por mes",
                    "filtros": {}
                }

        return pred

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
        pred = predictor_visual.predecir_bloque(parte) or {}
        
        # Corregir la predicción con heurísticas locales si estamos en modo offline
        if offline_mode:
            pred = corregir_prediccion_heuristica(parte, pred)

        if pred and pred.get("intencion"):
            titulo_bloque = pred.get("titulo", pred["intencion"].replace("_", " ").capitalize())
            bloques_detectados.append(BloqueReporteIntent(
                tipo=pred["tipo"],
                intencion=pred["intencion"],
                titulo=titulo_bloque,
                orden=orden,
                entidadPrincipal=pred["entidadPrincipal"],
                metrica=pred["metrica"],
                limite=10,
                filtros=pred.get("filtros", {})
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
        titulo="Reporte Inteligente",
        descripcion="Interpretado mediante modelo local Keras y heurísticas.",
        bloques=bloques_detectados
    )
