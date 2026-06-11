import random
import uuid
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

def generar_dataset_local(politica: Dict[str, Any], cantidad: int) -> List[Dict[str, Any]]:
    logger.info(f"Iniciando simulación local ultra-rápida de {cantidad} escenarios para la política {politica.get('id')}")
    
    nodos_dict = {str(n.get("id")): n for n in politica.get("nodos", [])}
    transiciones = politica.get("transiciones") or politica.get("conexiones") or []
    carriles = [c.get("nombre", "General") for c in politica.get("carriles", [{"nombre": "Sistema"}])]
    
    # Construir grafo
    edges = {}
    for t in transiciones:
        origen = str(t.get("origen"))
        edges.setdefault(origen, []).append(t)
        
    inicio_nodes = [str(n.get("id")) for n in politica.get("nodos", []) if n.get("tipo") in ["INICIO", "START_EVENT"]]
    
    if not inicio_nodes:
        if nodos_dict:
            inicio_nodes = [list(nodos_dict.keys())[0]]
        else:
            logger.warning("No hay nodos en la política.")
            return []
            
    escenarios = []
    
    for _ in range(cantidad):
        inicio = random.choice(inicio_nodes)
        current = inicio
        
        path = []
        path_ids = []
        
        if current in nodos_dict:
            path.append(nodos_dict[current].get("nombre", current))
        path_ids.append(current)
        
        visited_activities = set()
        carriles_visitados = set()
        decisiones = []
        
        nodos_count = 1
        decisiones_count = 0
        retornos_count = 0
        
        duracion_total = 0
        nodos_duraciones = {}
        
        # Simulamos la ejecución recorriendo el grafo
        while current in edges and edges[current]:
            # Detener si hay un loop infinito muy largo
            if nodos_count > 50:
                break
                
            node_info = nodos_dict.get(current, {})
            
            # Asignar un carril aleatorio o usar el que tiene
            carriles_visitados.add(random.choice(carriles))
            
            tipo = str(node_info.get("tipo", "")).upper()
            nombre_nodo = node_info.get("nombre", current)
            
            if tipo in ["ACTIVIDAD", "TAREA", "USER_TASK", "SERVICE_TASK"]:
                visited_activities.add(nombre_nodo)
                # Simular tiempo de actividad (ej: 10 a 120 mins)
                dur = random.randint(10, 120)
                duracion_total += dur
                nodos_duraciones[nombre_nodo] = nodos_duraciones.get(nombre_nodo, 0) + dur
                
            elif tipo in ["DECISION", "GATEWAY", "EXCLUSIVE_GATEWAY"]:
                decisiones_count += 1
                dur = random.randint(1, 15)
                duracion_total += dur
                nodos_duraciones[nombre_nodo] = nodos_duraciones.get(nombre_nodo, 0) + dur
            
            # Elegir siguiente transición
            opciones = edges[current]
            
            # Simular lógica: 80% camino principal, 20% retorno/error
            weights = []
            for op in opciones:
                if "condicion" in op and ("rechaz" in str(op["condicion"]).lower() or "retorno" in str(op["condicion"]).lower()):
                    weights.append(0.2)
                else:
                    weights.append(0.8)
                    
            next_trans = random.choices(opciones, weights=weights, k=1)[0]
            
            if "condicion" in next_trans and next_trans["condicion"]:
                decisiones.append(str(next_trans["condicion"]))
                
            next_node = str(next_trans.get("destino"))
            
            # Detectar si es un retorno (loop)
            if next_node in path_ids:
                retornos_count += 1
                
            path_ids.append(next_node)
            if next_node in nodos_dict:
                path.append(nodos_dict[next_node].get("nombre", next_node))
            else:
                path.append(next_node)
                
            current = next_node
            nodos_count += 1
            
        # Calcular cuellos de botella
        actividad_lenta = max(nodos_duraciones, key=nodos_duraciones.get) if nodos_duraciones else "Ninguna"
        
        # Etiquetado para Deep Learning
        riesgo_demora = 1 if (duracion_total > 300 or retornos_count >= 1) else 0
        cuello_botella = 1 if duracion_total > 400 else 0
        anomalia = 1 if retornos_count > 2 else 0
        
        prioridad_actual = random.choices(["BAJA", "NORMAL", "ALTA"], weights=[0.4, 0.5, 0.1])[0]
        prioridad_rec = "ALTA" if (riesgo_demora == 1 or retornos_count > 0) else prioridad_actual
        
        # Rutas completas
        ruta_str = "->".join(path_ids)
        ruta_legible = " -> ".join(path)
        
        # Ruta recomendada (simulamos la ruta más directa / "happy path")
        # Para simplificar, si hubo retornos, la recomendada sería la ejecutada sin retornos
        happy_path = [p for p in path_ids if path_ids.count(p) == 1]
        ruta_recomendada_label = "RUTA_DIRECTA" if retornos_count > 0 else "RUTA_ACTUAL"
        
        escenario = {
            "instanciaId": f"SIM-LOCAL-{uuid.uuid4().hex[:8]}",
            "politicaId": politica.get("id", "UNKNOWN"),
            "nombrePolitica": politica.get("nombre", "UNKNOWN"),
            "origenDatos": "SIMULADO_LOCAL",
            "rutaEjecutadaCodificada": ruta_str,
            "rutaEjecutadaLegible": ruta_legible,
            "carrilesVisitados": ", ".join(carriles_visitados) if carriles_visitados else "General",
            "actividadesVisitadas": ", ".join(visited_activities) if visited_activities else "Ninguna",
            "decisionesTomadas": ", ".join(decisiones) if decisiones else "Ninguna",
            "cantidadObservaciones": random.randint(1, 4) if retornos_count > 0 else 0,
            "cantidadNodos": nodos_count,
            "cantidadDecisiones": decisiones_count,
            "cantidadForks": 0,
            "cantidadJoins": 0,
            "cantidadRetornos": retornos_count,
            "cantidadReprocesos": retornos_count,
            "cantidadDocumentos": random.randint(1, 6),
            "cantidadFuncionariosInvolucrados": len(carriles_visitados) + random.randint(0, 2),
            "duracionPromedioHistorica": float(duracion_total),
            "nodoMasLento": actividad_lenta,
            "actividadMasLenta": actividad_lenta,
            "carrilMasLento": list(carriles_visitados)[0] if carriles_visitados else "General",
            "cuelloBotellaLabel": cuello_botella,
            "anomaliaLabel": anomalia,
            "riesgoDemoraLabel": riesgo_demora,
            "prioridadActual": prioridad_actual,
            "prioridadRecomendadaLabel": prioridad_rec,
            "rutaRecomendadaLabel": ruta_recomendada_label,
            "rutaRecomendadaLegible": " -> ".join([nodos_dict.get(n, {}).get("nombre", n) for n in happy_path]) if happy_path else ruta_legible,
            "motivoRecomendacion": "Generado localmente para volumen de DL."
        }
        
        escenarios.append(escenario)
        
    return escenarios
