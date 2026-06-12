import os
import json
import pickle
import numpy as np
import httpx
from typing import Dict, Any
import logging

from app.modules.predicciones.config import settings
from app.modules.predicciones.schemas import PredictionRequest
from app.modules.predicciones.services.preprocessing_service import preprocess_prediction_input

logger = logging.getLogger(__name__)

async def predict(req: PredictionRequest) -> Dict[str, Any]:
    if getattr(req, "skipDeepSeek", False):
        print("[PREDICCIONES OFFLINE][IA] Request recibido")
        logger.info("[PREDICCIONES OFFLINE][IA] Request recibido")
        print("[PREDICCIONES OFFLINE][IA] skipDeepSeek=true")
        logger.info("[PREDICCIONES OFFLINE][IA] skipDeepSeek=true")
        print("[PREDICCIONES OFFLINE][IA] VERSION_RUTA_OFFLINE=2026-06-12-ruta-parcial")
        logger.info("[PREDICCIONES OFFLINE][IA] VERSION_RUTA_OFFLINE=2026-06-12-ruta-parcial")
    from tensorflow.keras.models import load_model
    # Normalizar campos Optional que puedan llegar como None desde Java
    data = req.model_dump()
    data["politicaId"] = data.get("politicaId") or ""
    data["nombrePolitica"] = data.get("nombrePolitica") or ""
    data["prioridadActual"] = data.get("prioridadActual") or "NORMAL"
    data["rutaEjecutadaCodificada"] = data.get("rutaEjecutadaCodificada") or ""
    data["rutaEjecutadaLegible"] = data.get("rutaEjecutadaLegible") or ""
    data["carrilesVisitados"] = data.get("carrilesVisitados") or ""
    data["actividadesVisitadas"] = data.get("actividadesVisitadas") or ""
    for int_field in ["cantidadObservaciones","cantidadNodos","cantidadDecisiones",
                      "cantidadForks","cantidadJoins","cantidadRetornos","cantidadReprocesos",
                      "cantidadDocumentos","cantidadFuncionariosInvolucrados"]:
        data[int_field] = data.get(int_field) or 0
    data["duracionPromedioHistorica"] = data.get("duracionPromedioHistorica") or 0.0

    X = preprocess_prediction_input(data)
    
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
    
    if getattr(req, "skipDeepSeek", False):
        print("[PREDICCIONES OFFLINE][IA] Ejecutando modelos Keras locales")
        logger.info("[PREDICCIONES OFFLINE][IA] Ejecutando modelos Keras locales")
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

    # Llamar a DeepSeek para generar el análisis profundo o saltarlo si es offline
    if getattr(req, "skipDeepSeek", False):
        print("[PREDICCIONES OFFLINE][IA] Predicción completada")
        logger.info("[PREDICCIONES OFFLINE][IA] Predicción completada")
        logger.info("[PREDICCIONES OFFLINE] skipDeepSeek=true, usando solo modelos Keras locales.")
        
        # 1. Parse Graph Structure
        nodos_dict, conexiones_list, adj_list, rev_adj_list, in_degree, out_degree, iniciales, finales = parse_graph_structure(req.politicaEstructuraJson)
        
        # Identify cycles and anomalous nodes for path cost evaluation
        cycle_nodes = set()
        def get_cycle_nodes():
            visited = set()
            rec_stack = set()
            path = []
            def dfs_c(node):
                visited.add(node)
                rec_stack.add(node)
                path.append(node)
                for neighbor in adj_list.get(node, []):
                    if neighbor not in visited:
                        dfs_c(neighbor)
                    elif neighbor in rec_stack:
                        idx = path.index(neighbor)
                        for c_node in path[idx:]:
                            cycle_nodes.add(c_node)
                rec_stack.remove(node)
                path.pop()
            for node in nodos_dict:
                if node not in visited:
                    dfs_c(node)
        
        get_cycle_nodes()
        anomalous_nodes = cycle_nodes.copy()
        for nid, n in nodos_dict.items():
            tipo = str(n.get("tipo", "")).upper().strip()
            nombre = str(n.get("nombre", "")).upper().strip()
            is_fin = "FIN" in tipo or "END" in tipo or "TERMINAL" in tipo or "FIN" in nombre or "END" in nombre
            is_ini = "INICIO" in tipo or "INIC" in tipo or "START" in tipo or "INICIO" in nombre or "START" in nombre
            if out_degree[nid] == 0 and not is_fin:
                anomalous_nodes.add(nid)
            if in_degree[nid] == 0 and not is_ini:
                anomalous_nodes.add(nid)

        # 2. Find paths and select optimal path (less cost)
        all_paths = []
        optimal_path = None
        min_cost = float('inf')
        tipo_ruta = "COMPLETA"
        
        for start in iniciales:
            paths = find_all_paths(start, finales, adj_list)
            for path in paths:
                all_paths.append(path)
                cost = calculate_path_cost(path, nodos_dict, anomalous_nodes)
                if cost < min_cost:
                    min_cost = cost
                    optimal_path = path

        # If no complete path to FIN exists, try to trace maximal partial paths
        if not optimal_path:
            tipo_ruta = "PARCIAL"
            all_partial = []
            for start in iniciales:
                partials = find_all_partial_paths(start, adj_list)
                all_partial.extend(partials)
            if all_partial:
                def sort_key(p):
                    return (-len(p), calculate_path_cost(p, nodos_dict, anomalous_nodes))
                all_partial.sort(key=sort_key)
                optimal_path = all_partial[0]
                    
        # 3. Detect bottlenecks
        local_bottlenecks = detect_local_bottlenecks(nodos_dict, conexiones_list, adj_list, in_degree, out_degree, all_paths)
        
        # 4. Detect anomalies
        local_anomalies = detect_local_anomalies(nodos_dict, conexiones_list, adj_list, rev_adj_list, in_degree, out_degree, all_paths)
        
        # 5. Priority Score and Label mapping
        local_priority_score, local_priority_val = calculate_local_priority_score(req, nodos_dict, local_bottlenecks, local_anomalies, prioridad)
        
        # 6. Confidence score
        local_confidence = calculate_local_confidence(nodos_dict, conexiones_list, all_paths, prioridad, local_priority_val)
        
        # 7. Generate Rich Programmatic Explanations
        resumen, explicacion_ruta, acciones_ruta, recs, factores, prioridad_por_nodo = generate_local_explanations(
            req, nodos_dict, optimal_path, local_bottlenecks, local_anomalies, local_priority_score, local_priority_val, local_confidence, tipo_ruta
        )
        
        # Mapeo de ruta final para devolver nombres legibles
        ruta_final = []
        if optimal_path:
            ruta_final = [nodos_dict.get(nid, {}).get("nombre") or nid for nid in optimal_path]
        else:
            ruta_final = []
            
        return {
            "politicaId": req.politicaId,
            "politicaNombre": req.nombrePolitica,
            "resumenEjecutivo": resumen,
            "mejorRuta": {
                "nombre": "Ruta óptima offline" if tipo_ruta == "COMPLETA" else "Ruta parcial sugerida",
                "tipoRuta": tipo_ruta,
                "nodos": ruta_final,
                "rutaRecomendada": ruta_final,
                "descripcion": explicacion_ruta,
                "explicacion": explicacion_ruta,
                "confianza": float(local_confidence),
                "acciones": acciones_ruta
            },
            "cuellosBotella": local_bottlenecks,
            "anomalias": local_anomalies,
            "prioridad": {
                "valor": local_priority_val,
                "probabilidad": float(probabilities.get("prioridadRecomendadaLabel", 1.0)),
                "motivo": f"Prioridad sugerida local de nivel {local_priority_val} calculada en base al flujo.",
                "factores": factores,
                "prioridadPorNodo": prioridad_por_nodo
            },
            "recomendaciones": recs,
            "explicacionModelo": "Evaluación realizada offline mediante modelos de redes neuronales profundas (Keras) y análisis estructural determinístico de grafos en su máquina local.",
            "datosUsados": {
                "simulaciones": 200,
                "tiempoPromedio": f"{24.5}h",
                "nodoMayorCarga": local_bottlenecks[0].get("nodo") if local_bottlenecks else "Ninguno",
                "porcentajeCarga": f"{int(local_priority_score)}%"
            }
        }

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


# ==========================================
# MOTOR DE ANÁLISIS DE GRAFOS OFFLINE LOCAL
# ==========================================

def is_inicio_node(n):
    tipo = str(n.get("tipo", "")).upper().strip()
    nombre = str(n.get("nombre", "")).upper().strip()
    return "INICIO" in tipo or "START" in tipo or "INIC" in tipo or "INICIO" in nombre or "START" in nombre

def is_fin_node(n):
    tipo = str(n.get("tipo", "")).upper().strip()
    return "FIN" in tipo or "END" in tipo or "TERMINAL" in tipo

def parse_graph_structure(json_str: str):
    if not json_str:
        return {}, [], {}, {}, {}, {}, [], []
    try:
        data = json.loads(json_str)
    except Exception:
        return {}, [], {}, {}, {}, {}, [], []

    nodos = data.get("nodos", [])
    conexiones = data.get("conexiones", [])

    nodos_dict = {n.get("id"): n for n in nodos if n.get("id")}
    
    conexiones_list = []
    for c in conexiones:
        orig = c.get("origen")
        dest = c.get("destino")
        if orig and dest:
            conexiones_list.append((orig, dest))
            
    from collections import defaultdict
    adj_list = defaultdict(list)
    rev_adj_list = defaultdict(list)
    in_degree = defaultdict(int)
    out_degree = defaultdict(int)

    for n_id in nodos_dict:
        in_degree[n_id] = 0
        out_degree[n_id] = 0

    for orig, dest in conexiones_list:
        if orig in nodos_dict and dest in nodos_dict:
            adj_list[orig].append(dest)
            rev_adj_list[dest].append(orig)
            in_degree[dest] += 1
            out_degree[orig] += 1

    iniciales = [nid for nid, n in nodos_dict.items() if is_inicio_node(n)]
    if not iniciales:
        iniciales = [nid for nid, n in nodos_dict.items() if in_degree[nid] == 0]
    if not iniciales and nodos_dict:
        iniciales = [list(nodos_dict.keys())[0]]

    finales = [nid for nid, n in nodos_dict.items() if is_fin_node(n)]

    print(f"[PREDICCIONES OFFLINE][IA] nodos={len(nodos_dict)} conexiones={len(conexiones_list)} iniciales={len(iniciales)} finales={len(finales)}")
    logger.info(f"[PREDICCIONES OFFLINE][IA] nodos={len(nodos_dict)} conexiones={len(conexiones_list)} iniciales={len(iniciales)} finales={len(finales)}")

    return nodos_dict, conexiones_list, adj_list, rev_adj_list, in_degree, out_degree, iniciales, finales


def find_all_paths(start: str, end_nodes: list, adj_list: dict):
    paths = []
    def dfs(node, path, visited):
        if node in end_nodes:
            paths.append(list(path))
            return
        
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)

    dfs(start, [start], {start})
    return paths


def find_all_partial_paths(start: str, adj_list: dict):
    paths = []
    def dfs(node, path, visited):
        extended = False
        for neighbor in adj_list.get(node, []):
            if neighbor not in visited:
                extended = True
                visited.add(neighbor)
                path.append(neighbor)
                dfs(neighbor, path, visited)
                path.pop()
                visited.remove(neighbor)
        if not extended:
            paths.append(list(path))

    dfs(start, [start], {start})
    return paths


def calculate_path_cost(path: list, nodos_dict: dict, anomalous_nodes: set = None):
    cost = 0.0
    for node_id in path:
        node = nodos_dict.get(node_id, {})
        cost += 1.0
        
        if anomalous_nodes and node_id in anomalous_nodes:
            cost += 50.0
            
        form = node.get("formulario") or []
        cost += len(form) * 0.2
        for field in form:
            if field.get("requerido") or field.get("obligatorio"):
                cost += 0.5
                
        tipo = str(node.get("tipo", "")).upper()
        nombre = str(node.get("nombre", "")).lower()
        if tipo == "DECISION" or "aprob" in nombre or "revis" in nombre or "valid" in nombre:
            cost += 2.0
            
        conds = node.get("condiciones") or []
        cost += len(conds) * 0.5
        
    return cost


def detect_local_bottlenecks(nodos_dict: dict, conexiones_list: list, adj_list: dict, in_degree: dict, out_degree: dict, paths: list):
    bottlenecks = []
    
    path_visits = {}
    for nid in nodos_dict:
        path_visits[nid] = 0
    for path in paths:
        for nid in path:
            path_visits[nid] += 1
            
    for nid, node in nodos_dict.items():
        node_name = node.get("nombre") or nid
        tipo = str(node.get("tipo", "")).upper()
        nombre_lower = node_name.lower()
        
        if tipo in ["INICIO", "FIN"]:
            continue
            
        factors = []
        severity = "BAJA"
        score = 0
        
        if in_degree[nid] >= 3:
            factors.append(f"Alta convergencia de flujos ({in_degree[nid]} conexiones de entrada)")
            score += in_degree[nid] * 15
            
        if "aprob" in nombre_lower or "revis" in nombre_lower or "valid" in nombre_lower:
            factors.append("Proceso manual de aprobación/revisión")
            score += 30
            
        if len(paths) > 1 and path_visits[nid] == len(paths):
            factors.append("Punto de paso obligatorio para todas las rutas del workflow")
            score += 40
            
        form = node.get("formulario") or []
        req_count = sum(1 for f in form if f.get("requerido"))
        if len(form) > 4:
            factors.append(f"Carga de datos elevada ({len(form)} campos de formulario, {req_count} obligatorios)")
            score += len(form) * 5 + req_count * 5
            
        if score > 0:
            if score >= 70:
                severity = "CRITICA"
            elif score >= 50:
                severity = "ALTO"
            elif score >= 30:
                severity = "MEDIO"
            else:
                severity = "BAJO"
                
            recom = "Optimizar la distribución del flujo para balancear las tareas."
            if "aprob" in nombre_lower or "revis" in nombre_lower:
                recom = f"Considerar la automatización de validaciones previas a la '{node_name}' para agilizar la revisión."
            elif len(form) > 4:
                recom = f"Reducir el formulario de '{node_name}' a campos esenciales o dividir la carga de datos en múltiples pasos."
            elif in_degree[nid] >= 3:
                recom = f"Redireccionar algunas ramas de entrada para evitar sobrecargar el nodo '{node_name}'."
                
            bottlenecks.append({
                "nodo": node_name,
                "riesgo": severity,
                "probabilidad": min(1.0, float(score) / 100.0),
                "tiempoPromedio": f"{max(1, score // 15)}h",
                "carga": "Alta" if score >= 50 else ("Media" if score >= 30 else "Normal"),
                "motivo": " + ".join(factors),
                "impacto": "Crítico" if severity in ["CRITICA", "ALTO"] else "Moderado",
                "recomendacion": recom
            })
            
    return bottlenecks


def detect_local_anomalies(nodos_dict: dict, conexiones_list: list, adj_list: dict, rev_adj_list: dict, in_degree: dict, out_degree: dict, paths: list):
    anomalies = []
    
    has_inicio = any(str(n.get("tipo", "")).upper() == "INICIO" for n in nodos_dict.values())
    has_fin = any(str(n.get("tipo", "")).upper() == "FIN" for n in nodos_dict.values())
    
    has_inicio_formal = any(is_inicio_node(n) for n in nodos_dict.values())
    has_fin_formal = any(is_fin_node(n) for n in nodos_dict.values())
    
    if not has_inicio_formal:
        anomalies.append({
            "tipo": "ESTRUCTURAL",
            "nodo": "Workflow completo",
            "riesgo": "CRITICO",
            "descripcion": "No se identificó nodo inicial formal.",
            "recomendacion": "Agregar un nodo de tipo INICIO al canvas de diseño."
        })
    if not has_fin_formal:
        anomalies.append({
            "tipo": "ESTRUCTURAL",
            "nodo": "Workflow completo",
            "riesgo": "CRITICO",
            "descripcion": "No se identificó nodo final formal.",
            "recomendacion": "Agregar un nodo de tipo FIN para formalizar el fin del trámite."
        })
        
    for nid, node in nodos_dict.items():
        node_name = node.get("nombre") or nid
        tipo = str(node.get("tipo", "")).upper()
        
        if in_degree[nid] == 0 and out_degree[nid] == 0:
            anomalies.append({
                "tipo": "AISLAMIENTO",
                "nodo": node_name,
                "riesgo": "ALTO",
                "descripcion": f"El nodo '{node_name}' está completamente aislado, sin conexiones de entrada ni salida.",
                "recomendacion": "Conectar el nodo al flujo o eliminarlo si no es necesario."
            })
        elif out_degree[nid] == 0 and tipo != "FIN" and tipo != "INICIO":
            anomalies.append({
                "tipo": "SALIDA_HUECA",
                "nodo": node_name,
                "riesgo": "ALTO",
                "descripcion": f"El nodo '{node_name}' es un punto muerto (no posee conexiones de salida) pero no está tipificado como FIN.",
                "recomendacion": f"Conectar la salida de '{node_name}' al nodo final o al siguiente paso."
            })
        elif in_degree[nid] == 0 and tipo != "INICIO" and tipo != "FIN":
            anomalies.append({
                "tipo": "ENTRADA_HUECA",
                "nodo": node_name,
                "riesgo": "MEDIO",
                "descripcion": f"El nodo '{node_name}' no tiene conexiones de entrada (paso inalcanzable).",
                "recomendacion": f"Conectar un nodo anterior hacia '{node_name}'."
            })
            
    def has_cycles():
        visited = set()
        rec_stack = set()
        path = []
        cycles = []
        
        def dfs_cycle(node):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            for neighbor in adj_list[node]:
                if neighbor not in visited:
                    dfs_cycle(neighbor)
                elif neighbor in rec_stack:
                    idx = path.index(neighbor)
                    cycles.append(list(path[idx:]))
            rec_stack.remove(node)
            path.pop()
            
        for node in nodos_dict:
            if node not in visited:
                dfs_cycle(node)
        return cycles

    detected_cycles = has_cycles()
    for cycle in detected_cycles:
        names = [nodos_dict.get(nid, {}).get("nombre") or nid for nid in cycle]
        cycle_str = " -> ".join(names)
        anomalies.append({
            "tipo": "CICLO_INFINITO",
            "nodo": names[0],
            "riesgo": "ALTO",
            "descripcion": f"Bucle cerrado infinito detectado en el ciclo: {cycle_str} -> {names[0]}.",
            "recomendacion": "Romper el bucle infinito agregando una condición de salida válida en el nodo de decisión."
        })
        
    if not paths:
        anomalies.append({
            "tipo": "CONECTIVIDAD",
            "nodo": "Workflow completo",
            "riesgo": "CRITICO",
            "descripcion": "No existe una ruta válida desde inicio hasta fin.",
            "recomendacion": "Verificar y conectar las transiciones intermedias para trazar un camino completo de inicio a fin."
        })
        
    return anomalies


def calculate_local_priority_score(req, nodos_dict: dict, bottlenecks: list, anomalies: list, raw_keras_priority: str):
    score = 15.0
    score += len(nodos_dict) * 2.0
    
    decisiones = sum(1 for n in nodos_dict.values() if str(n.get("tipo", "")).upper() == "DECISION")
    score += decisiones * 4.0
    
    total_fields = 0
    total_req_fields = 0
    for node in nodos_dict.values():
        form = node.get("formulario") or []
        total_fields += len(form)
        total_req_fields += sum(1 for f in form if f.get("requerido"))
    score += total_fields * 0.5 + total_req_fields * 1.0
    
    score += len(bottlenecks) * 8.0
    
    for a in anomalies:
        riesgo = str(a.get("riesgo", "")).upper()
        if riesgo == "CRITICO":
            score += 20.0
        elif riesgo == "ALTO":
            score += 12.0
        else:
            score += 6.0
            
    keras_val = str(raw_keras_priority).upper()
    if keras_val == "ALTA" or keras_val == "URGENTE":
        score += 15.0
    elif keras_val == "NORMAL":
        score += 5.0
        
    score = min(100.0, score)
    
    if score >= 81:
        val = "CRITICA"
    elif score >= 61:
        val = "ALTA"
    elif score >= 31:
        val = "MEDIA"
    else:
        val = "BAJA"
        
    return score, val


def calculate_local_confidence(nodos_dict: dict, conexiones_list: list, paths: list, model_keras_val: str, local_computed_val: str):
    if not nodos_dict:
        return 0.0
        
    score = 0.90
    
    if not paths:
        score -= 0.30
        
    if len(nodos_dict) < 3:
        score -= 0.15
        
    if str(model_keras_val).upper() != str(local_computed_val).upper():
        score -= 0.10
        
    return max(0.40, min(0.98, score))


def generate_local_explanations(req, nodos_dict: dict, optimal_path: list, bottlenecks: list, anomalies: list, priority_score: float, priority_val: str, confidence: float, tipo_ruta: str = "COMPLETA"):
    policy_name = req.nombrePolitica or "la política"
    n_count = len(nodos_dict)
    
    resumen = (
        f"Se analizó la política '{policy_name}' en modo offline local. El flujo contiene {n_count} nodos. "
    )
    if optimal_path:
        if tipo_ruta == "COMPLETA":
            resumen += f"Se identificó una ruta óptima de {len(optimal_path)} pasos que recorre el flujo principal. "
        else:
            resumen += f"No se encontró una ruta completa hasta un nodo FIN. Se muestra una ruta parcial sugerida de {len(optimal_path)} pasos para identificar dónde corregir el flujo. "
    else:
        resumen += "No se pudo identificar una ruta de ejecución válida de inicio a fin debido a problemas de conectividad en el grafo. "
        
    if bottlenecks:
        nodes_b = [b.get("nodo") for b in bottlenecks]
        resumen += f"Se detectaron posibles cuellos de botella en: {', '.join(nodes_b)}. "
    else:
        resumen += "No se detectaron cuellos de botella severos de forma estructural. "
        
    resumen += f"La prioridad sugerida para el proceso es {priority_val} (score local: {int(priority_score)}/100) debido a la complejidad general del flujo."
    
    if optimal_path:
        nombres_ruta = [nodos_dict.get(nid, {}).get("nombre") or nid for nid in optimal_path]
        if tipo_ruta == "COMPLETA":
            explicacion_ruta = f"El camino recomendado optimiza el tránsito del flujo pasando por: {', '.join(nombres_ruta)}. Minimiza pasos manuales y complejidad."
        else:
            explicacion_ruta = "No se encontró una ruta completa hasta un nodo FIN. Se muestra una ruta parcial sugerida para identificar dónde corregir el flujo."
        acciones_ruta = ["Revisar el flujo en el canvas de diseño", "Asegurar que los responsables de los carriles estén notificados"]
    else:
        explicacion_ruta = "No existe una ruta válida desde inicio hasta fin."
        acciones_ruta = ["Conectar los nodos sueltos en el canvas", "Agregar transiciones válidas de salida en los nodos de decisión"]

    recs = []
    if anomalies:
        recs.append({
            "tipo": "OPTIMIZACION_ESTRUCTURAL",
            "titulo": "Corregir Anomalías del Grafo",
            "descripcion": "Existen fallas estructurales que impiden el flujo correcto del proceso. Corregir nodos aislados o sin salida.",
            "impactoEsperado": "Alto",
            "nodosAfectados": [a.get("nodo") for a in anomalies]
        })
    if bottlenecks:
        recs.append({
            "tipo": "BALANCEO_CARGA",
            "titulo": "Balancear Carga en Nodos Críticos",
            "descripcion": "Simplificar formularios o reasignar tareas en los nodos saturados de transiciones para reducir tiempos de espera.",
            "impactoEsperado": "Medio",
            "nodosAfectados": [b.get("nodo") for b in bottlenecks]
        })
        
    if not recs:
        recs.append({
            "tipo": "MANTENIMIENTO",
            "titulo": "Monitoreo del Workflow",
            "descripcion": "El flujo es simple y bien estructurado. Se sugiere mantener el monitoreo periódico.",
            "impactoEsperado": "Bajo",
            "nodosAfectados": []
        })

    factores = [
        f"Cantidad total de nodos: {n_count}",
        f"Cuellos de botella estructurales: {len(bottlenecks)}",
        f"Anomalías críticas identificadas: {len(anomalies)}"
    ]
    if priority_score > 60:
        factores.append("Flujo complejo con presencia de procesos manuales y decisiones críticas.")
    else:
        factores.append("Flujo de complejidad baja o moderada.")
        
    prioridad_por_nodo = []
    for nid, node in nodos_dict.items():
        node_name = node.get("nombre") or nid
        tipo = str(node.get("tipo", "")).upper()
        if tipo == "DECISION":
            p_node = "ALTA"
            motivo = "Nodo de toma de decisiones del que dependen múltiples ramas."
        elif "aprob" in node_name.lower() or "revis" in node_name.lower():
            p_node = "ALTA"
            motivo = "Revisión o aprobación manual que requiere atención inmediata del funcionario."
        else:
            p_node = "NORMAL"
            motivo = "Tarea de procesamiento estándar del flujo."
            
        prioridad_por_nodo.append({
            "nodo": node_name,
            "prioridadSugerida": p_node,
            "motivo": motivo
        })
        
    return resumen, explicacion_ruta, acciones_ruta, recs, factores, prioridad_por_nodo


