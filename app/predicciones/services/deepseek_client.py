import httpx
import json
import logging
from typing import List, Dict, Any
from app.modules.predicciones.config import settings
from app.modules.predicciones.schemas import SyntheticScenario, DeepSeekResponse

logger = logging.getLogger(__name__)

DEEPSEEK_PROMPT_TEMPLATE = """
Eres un experto en procesos de negocio BPMN y generador de datos sintéticos masivos para entrenar Redes Neuronales.
Basado en la siguiente estructura de politica de negocio real, genera {cantidad} escenarios simulados altamente ricos y variados.
IMPORTANTE: RESPONDE ÚNICAMENTE CON JSON VÁLIDO. NO USES MARKDOWN. NO INCLUYAS NINGÚN TEXTO ADICIONAL ANTES O DESPUÉS DEL JSON.

Estructura de la politica:
{estructura_politica}

Reglas de Negocio para Máxima Variabilidad (¡HAZLOS MUY DISTINTOS ENTRE SÍ!):
1. **'decisionesTomadas'**: NO uses 'Ninguna'. Inventa y escribe las condiciones lógicas evaluadas. Ej: 'Documentación completa=NO|Corrección recibida=SI'.
2. **'duracionPromedioHistorica'**: Genera números realistas que varíen entre flujos muy rápidos (100) y extremos (6000) dependiendo de los retornos.
3. **Labels**: Si generas retornos/reprocesos, el `riesgoDemoraLabel` debe ser 1 y `prioridadRecomendadaLabel` debe ser ALTA.
4. **Rutas Ejecutadas**: Genera 'happy paths' (rutas directas), pero también simula un 40% de rutas con múltiples rechazos, correcciones o cuellos de botella (donde `cuelloBotellaLabel`=1).
5. **'motivoRecomendacion'**: Escribe razones largas y descriptivas como: "Se recomienda la ruta directa porque históricamente reduce retornos. Posible cuello de botella en Admisión."
6. **'actividadesVisitadas' y 'carrilesVisitados'**: Pon siempre los nombres separados por comas según el recorrido simulado, NUNCA "Ninguna".

El formato del JSON debe ser exactamente:
{{
  "escenarios": [
    {{
      "instanciaId": "SIM-...",
      "politicaId": "...",
      "nombrePolitica": "...",
      "origenDatos": "SIMULADO_DEEPSEEK",
      "rutaEjecutadaCodificada": "...",
      "rutaEjecutadaLegible": "...",
      "carrilesVisitados": "...",
      "actividadesVisitadas": "...",
      "decisionesTomadas": "...",
      "cantidadObservaciones": 0,
      "cantidadNodos": 0,
      "cantidadDecisiones": 0,
      "cantidadForks": 0,
      "cantidadJoins": 0,
      "cantidadRetornos": 0,
      "cantidadReprocesos": 0,
      "cantidadDocumentos": 0,
      "cantidadFuncionariosInvolucrados": 0,
      "duracionPromedioHistorica": 0,
      "nodoMasLento": "...",
      "actividadMasLenta": "...",
      "carrilMasLento": "...",
      "cuelloBotellaLabel": 0,
      "anomaliaLabel": 0,
      "riesgoDemoraLabel": 0,
      "prioridadActual": "NORMAL",
      "prioridadRecomendadaLabel": "NORMAL",
      "rutaRecomendadaLabel": "...",
      "rutaRecomendadaLegible": "...",
      "motivoRecomendacion": "..."
    }}
  ]
}}
"""

async def generar_escenarios_deepseek(politica: Dict[str, Any], cantidad: int) -> Dict[str, Any]:
    prompt = DEEPSEEK_PROMPT_TEMPLATE.format(
        cantidad=cantidad,
        estructura_politica=json.dumps(politica, ensure_ascii=False, indent=2)
    )
    
    debug_info = {
        "deepseek_llamado": True,
        "status_code_deepseek": None,
        "error_deepseek": None,
        "respuesta_cruda_preview": None,
        "motivos_descarte": []
    }
    
    api_key_preview = settings.DEEPSEEK_API_KEY[:4] + "***" if settings.DEEPSEEK_API_KEY else "None"
    logger.info(f"Llamando a DeepSeek en {settings.DEEPSEEK_BASE_URL} con API KEY: {api_key_preview}")
    
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You are a data generator that strictly outputs valid JSON only. Do not wrap in markdown tags like ```json."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    logger.info(f"Payload enviado a DeepSeek: {json.dumps(payload, ensure_ascii=False)[:200]}...")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{settings.DEEPSEEK_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60)
            debug_info["status_code_deepseek"] = response.status_code
            response.raise_for_status()
            
            response_data = response.json()
            content = response_data["choices"][0]["message"]["content"]
            
            debug_info["respuesta_cruda_preview"] = content[:500] + "..." if len(content) > 500 else content
            logger.info("Respuesta cruda de DeepSeek recibida.")
            
            # Clean possible markdown block
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            elif content.startswith("```"):
                content = content[3:]
                
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            try:
                parsed_json = json.loads(content)
                escenarios = parsed_json.get("escenarios", [])
                if not escenarios:
                    debug_info["motivos_descarte"].append("El JSON parseado no contiene la clave 'escenarios' o la lista está vacía.")
                return {"escenarios": escenarios, "debug": debug_info}
            except json.JSONDecodeError as je:
                debug_info["error_deepseek"] = f"Error parseando JSON: {str(je)}"
                debug_info["motivos_descarte"].append("Respuesta no es JSON válido.")
                logger.error(f"Error parseando JSON de DeepSeek: {je}. Raw: {content}")
                return {"escenarios": [], "debug": debug_info}
                
    except httpx.HTTPStatusError as e:
        debug_info["error_deepseek"] = f"HTTP Error: {e.response.text}"
        logger.error(f"HTTPStatusError calling DeepSeek: {e.response.text}")
        return {"escenarios": [], "debug": debug_info}
    except Exception as e:
        debug_info["error_deepseek"] = f"Exception: {str(e)}"
        logger.error(f"Error calling DeepSeek for data generation: {e}")
        return {"escenarios": [], "debug": debug_info}

async def explicar_prediccion(prediccion: Dict[str, Any]) -> str:
    prompt = f"""
    Eres un asistente de IA para explicar resultados de un modelo de Deep Learning en un sistema de workflows (BPM).
    El modelo ha predecido lo siguiente para un trámite:
    {json.dumps(prediccion, ensure_ascii=False, indent=2)}
    
    Genera una explicación en lenguaje natural (máximo 3 oraciones) dirigida a un administrador/funcionario.
    IMPORTANTE: No digas que la IA modificó el workflow. Debes decir que esto es solo una recomendación o una predicción.
    """
    
    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "Eres un asistente útil y profesional."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{settings.DEEPSEEK_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            response_data = response.json()
            return response_data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        logger.error(f"Error calling DeepSeek for explanation: {e}")
        return "No se pudo generar una explicación con DeepSeek."
