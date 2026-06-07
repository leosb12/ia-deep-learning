import os
import json
import pickle
import numpy as np
import httpx
from tensorflow.keras.models import load_model
from typing import Dict, Any
import logging

from app.predicciones.config import settings
from app.predicciones.schemas import PredictionRequest
from app.predicciones.services.preprocessing_service import preprocess_prediction_input

logger = logging.getLogger(__name__)

async def predict(req: PredictionRequest) -> Dict[str, Any]:
    X = preprocess_prediction_input(req.model_dump())
    
    metadata_path = os.path.join(settings.MODEL_DIR, "training_metadata.json")
    if not os.path.exists(metadata_path):
        raise ValueError("Metadata de entrenamiento no encontrada. Ejecuta /train primero.")
        
    with open(metadata_path, 'r', encoding='utf-8') as f:
        models_info = json.load(f)
        
    try:
        with open(os.path.join(settings.MODEL_DIR, 'label_encoders.pkl'), 'rb') as f:
            label_encoders = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Encoders no encontrados.")

    results = {}
    probabilities = {}
    
    for label_name, info in models_info.items():
        le = label_encoders.get(label_name)
        if not info.get("model_path"):
            if le and len(le.classes_) > 0:
                results[label_name] = le.classes_[0]
                probabilities[label_name] = 1.0
            else:
                results[label_name] = "N/A"
                probabilities[label_name] = 0.0
            continue
            
        model = load_model(info["model_path"])
        preds = model.predict(X, verbose=0)[0]
        
        num_classes = info["classes"]
        if num_classes == 2:
            prob = float(preds[0])
            predicted_class_idx = 1 if prob > 0.5 else 0
            prob_final = prob if predicted_class_idx == 1 else (1.0 - prob)
            results[label_name] = le.inverse_transform([predicted_class_idx])[0]
            probabilities[label_name] = prob_final
        else:
            predicted_class_idx = np.argmax(preds)
            prob_final = float(preds[predicted_class_idx])
            results[label_name] = le.inverse_transform([predicted_class_idx])[0]
            probabilities[label_name] = prob_final

    def format_binary(val):
        return "SI" if str(val) == "1" else ("NO" if str(val) == "0" else str(val))
        
    riesgo = format_binary(results.get("riesgoDemoraLabel", "0"))
    cuello = format_binary(results.get("cuelloBotellaLabel", "0"))
    anomalia = format_binary(results.get("anomaliaLabel", "0"))
    prioridad = results.get("prioridadRecomendadaLabel", req.prioridadActual)
    ruta = results.get("rutaRecomendadaLabel", "N/A")

    # Llamar a DeepSeek para generar el análisis profundo
    return await analizar_con_deepseek(req, results, probabilities, riesgo, cuello, anomalia, prioridad, ruta)


def analyze_graph_structure(json_str):
    try:
        data = json.loads(json_str)
        nodos = {n.get('id'): n for n in data.get('nodos', [])}
        conexiones = data.get('conexiones', [])
        from collections import defaultdict
        adj = defaultdict(list)
        for c in conexiones:
            if 'origen' in c and 'destino' in c:
                adj[c['origen']].append(c['destino'])
                
        has_fin = any(n.get('tipo', '').upper() == 'FIN' for n in nodos.values())
        
        def has_cycle():
            visited = set()
            rec_stack = set()
            def dfs(node, path):
                visited.add(node)
                rec_stack.add(node)
                path.append(node)
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        cycle = dfs(neighbor, path)
                        if cycle: return cycle
                    elif neighbor in rec_stack:
                        idx = path.index(neighbor)
                        return path[idx:]
                rec_stack.remove(node)
                path.pop()
                return None
            for node in nodos:
                if node not in visited:
                    cycle = dfs(node, [])
                    if cycle: return cycle
            return None
            
        cycle_nodes = has_cycle()
        anomalias = []
        if not has_fin and len(nodos) > 0:
            anomalias.append("Falta nodo FIN (el proceso no tiene punto de finalización formal)")
        if cycle_nodes:
            names = [nodos.get(nid, {}).get('nombre', nid) for nid in cycle_nodes]
            anomalias.append(f"Bucle infinito (ciclo cerrado) entre: {' -> '.join(names)}")
            
        return anomalias
    except Exception as e:
        return []

async def analizar_con_deepseek(req: PredictionRequest, results: Dict, probs: Dict, riesgo: str, cuello: str, anomalia: str, prioridad: str, ruta: str) -> Dict[str, Any]:
    if not settings.DEEPSEEK_API_KEY:
        # Fallback if no API KEY
        return {
            "politicaId": req.politicaId,
            "politicaNombre": req.nombrePolitica,
            "resumenEjecutivo": "Motor DeepSeek no configurado. Resultados básicos.",
            "mejorRuta": {"rutaRecomendada": [ruta], "explicacion": "...", "confianza": probs.get("rutaRecomendadaLabel", 0.0), "acciones": []},
            "cuellosBotella": [], "anomalias": [], "prioridad": {"valor": prioridad, "probabilidad": probs.get("prioridadRecomendadaLabel", 0.0), "motivo": "", "factores": []},
            "recomendaciones": [], "explicacionModelo": "", "datosUsados": {}
        }

    politica_json = req.politicaEstructuraJson if req.politicaEstructuraJson else "{}"
    
    estructurales = analyze_graph_structure(politica_json)
    alertas_extra = ""
    if estructurales:
        alertas_extra = "\\n¡ALERTA CRÍTICA DEL SISTEMA DETERMINÍSTICO!: Se han detectado los siguientes errores estructurales matemáticos en el grafo:\\n" + "\\n".join(["- " + a for a in estructurales]) + "\\nDEBES REGISTRAR ESTAS ANOMALÍAS EN TU JSON SÍ O SÍ CON RIESGO 'CRITICO'."
    
    prompt = f"""
Eres un motor de Análisis Predictivo Inteligente (Deep Learning y NLP) para workflows BPM.
Se ha evaluado la política '{req.nombrePolitica}' (ID: {req.politicaId}).
El modelo de Deep Learning (Keras) ya ha dado sus predicciones crudas basándose en datos históricos. Tu tarea es EXPLICAR Y AMPLIAR estos resultados usando la estructura real de la política.

PREDICCIONES CRUDAS DEL MODELO KERAS:
- Cuello de Botella: {cuello} (Confianza: {probs.get('cuelloBotellaLabel', 0.0):.2f})
- Anomalía: {anomalia} (Confianza: {probs.get('anomaliaLabel', 0.0):.2f})
- Riesgo Demora: {riesgo} (Confianza: {probs.get('riesgoDemoraLabel', 0.0):.2f})
- Prioridad: {prioridad} (Confianza: {probs.get('prioridadRecomendadaLabel', 0.0):.2f})
- Ruta Recomendada Code: {ruta} (Confianza: {probs.get('rutaRecomendadaLabel', 0.0):.2f})


ESTRUCTURA DE LA POLÍTICA (JSON):
{politica_json}
{alertas_extra}

INSTRUCCIONES:

1. Genera un análisis ejecutivo súper profesional y extenso para un gerente/administrador.
2. Extrae los nombres reales de los nodos del JSON de la política. NO inventes nombres.
3. IMPORTANTE SOBRE CUELLOS DE BOTELLA Y ANOMALÍAS: El modelo Keras evaluó métricas globales sin entender la topología del grafo. TÚ debes analizar las conexiones del JSON. Si detectas un bucle infinito (ej. A -> B -> A sin salida), nodos aislados, o tareas sin conexión al 'Fin', DEBES registrarlo como una "Anomalía" grave con alta probabilidad (ej. 0.95+), INCLUSO SI KERAS DIJO 'NO'.
4. Si ves que múltiples tareas convergen en una sola, o hay tareas típicamente lentas (Revisiones, Aprobaciones), regístralas como "Cuellos de Botella" con alta probabilidad, contradiciendo a Keras si es necesario.
5. Justifica las predicciones usando lógica de BPM.
6. Genera al menos 2 recomendaciones accionables de optimización del flujo.
7. Para la 'Mejor Ruta', evalúa la complejidad y asigna una 'confianza' realista.
8. Para la 'Prioridad', analiza cada nodo y asigna una prioridad específica (ALTA, MEDIA, BAJA) para el funcionario que lo ejecutará, con un motivo breve.
9. RESPONDE ÚNICAMENTE CON UN JSON VÁLIDO. SIN MARKDOWN.

EL JSON DEBE TENER ESTA ESTRUCTURA EXACTA:
{{
  "politicaId": "{req.politicaId}",
  "politicaNombre": "{req.nombrePolitica}",
  "resumenEjecutivo": "Un párrafo extenso resumiendo...",
  "mejorRuta": {{
    "rutaRecomendada": ["Inicio", "Nodo 1", "Fin"],
    "explicacion": "...",
    "confianza": 0.88,
    "acciones": ["..."]
  }},
  "cuellosBotella": [
    {{
      "nodo": "...",
      "riesgo": "ALTO",
      "probabilidad": {probs.get('cuelloBotellaLabel', 0.0):.2f},
      "tiempoPromedio": "...",
      "carga": "...",
      "motivo": "...",
      "impacto": "...",
      "recomendacion": "..."
    }}
  ],
  "anomalias": [
    {{
      "tipo": "...",
      "nodo": "...",
      "riesgo": "...",
      "descripcion": "...",
      "recomendacion": "..."
    }}
  ],
  "prioridad": {{
    "valor": "{prioridad}",
    "probabilidad": {probs.get('prioridadRecomendadaLabel', 0.0):.2f},
    "motivo": "...",
    "factores": ["..."],
    "prioridadPorNodo": [
      {{
        "nodo": "Nombre del nodo real",
        "prioridadSugerida": "ALTA",
        "motivo": "Explicación de por qué el funcionario debe priorizar esto"
      }}
    ]
  }},
  "recomendaciones": [
    {{
      "tipo": "OPTIMIZACION_FLUJO",
      "titulo": "...",
      "descripcion": "...",
      "impactoEsperado": "...",
      "nodosAfectados": ["..."]
    }}
  ],
  "explicacionModelo": "...",
  "datosUsados": {{
    "simulaciones": 250,
    "tiempoPromedio": "...",
    "nodoMayorCarga": "...",
    "porcentajeCarga": "..."
  }}
}}
"""

    headers = {
        "Authorization": f"Bearer {settings.DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "You output JSON only. Do not wrap in markdown tags like ```json."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{settings.DEEPSEEK_BASE_URL}/chat/completions", json=payload, headers=headers, timeout=60)
            response.raise_for_status()
            
            content = response.json()["choices"][0]["message"]["content"].strip()
            if content.startswith("```json"): content = content[7:]
            elif content.startswith("```"): content = content[3:]
            if content.endswith("```"): content = content[:-3]
            content = content.strip()
            
            return json.loads(content)
    except Exception as e:
        logger.error(f"Error llamando a DeepSeek en predict: {e}")
        # Retornar un JSON de fallback
        return {
            "politicaId": req.politicaId,
            "politicaNombre": req.nombrePolitica,
            "resumenEjecutivo": "Se detectó un error al generar la explicación con el modelo de lenguaje. A continuación los datos crudos.",
            "mejorRuta": {"rutaRecomendada": [ruta], "explicacion": "Predicción cruda.", "confianza": probs.get("rutaRecomendadaLabel", 0.0), "acciones": []},
            "cuellosBotella": [], "anomalias": [], "prioridad": {"valor": prioridad, "probabilidad": probs.get("prioridadRecomendadaLabel", 0.0), "motivo": "Error", "factores": []},
            "recomendaciones": [], "explicacionModelo": str(e), "datosUsados": {}
        }

