import random
import uuid
import logging
import asyncio
from app.modules.predicciones.services.backend_client import get_politicas_reales
from app.modules.predicciones.services.dataset_storage_service import save_synthetic_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Diccionario para mapear IDs de departamento a nombres simulados reales
DEPARTAMENTOS = {
    "69e077f81052ca0aedcc381a": "Presupuesto",
    "69e1ce387696f1128cf873c4": "Dirección",
    "69e1df617696f1128cf873c5": "Admisión",
    "default": "Operaciones"
}

def generar_dataset_avanzado(politica, cantidad):
    nodos_dict = {str(n.get("id")): n for n in politica.get("nodos", [])}
    # La API devuelve 'conexiones', no 'transiciones'
    conexiones = politica.get("conexiones", [])
    
    edges = {}
    for t in conexiones:
        origen = str(t.get("origen"))
        edges.setdefault(origen, []).append(t)
        
    inicio_nodes = [str(n.get("id")) for n in politica.get("nodos", []) if n.get("tipo") in ["INICIO", "START_EVENT"]]
    if not inicio_nodes:
        if nodos_dict:
            inicio_nodes = [list(nodos_dict.keys())[0]]
        else:
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
        carriles_visitados = []
        decisiones = []
        
        nodos_count = 1
        decisiones_count = 0
        retornos_count = 0
        forks = 0
        joins = 0
        
        duracion_total = 0
        nodos_duraciones = {}
        
        while current in edges and edges[current]:
            if nodos_count > 40:
                break
                
            node_info = nodos_dict.get(current, {})
            tipo = str(node_info.get("tipo", "")).upper()
            nombre_nodo = node_info.get("nombre", current)
            
            # Extraer carril/departamento real
            dep_id = node_info.get("departamentoId")
            if dep_id:
                carril = DEPARTAMENTOS.get(dep_id, f"Depto-{str(dep_id)[:4]}")
            else:
                carril = "General"
                
            if carril not in carriles_visitados:
                carriles_visitados.append(carril)
                
            # Calcular tiempos
            if tipo in ["ACTIVIDAD", "TAREA"]:
                visited_activities.add(nombre_nodo)
                dur = random.randint(30, 240)
                duracion_total += dur
                nodos_duraciones[nombre_nodo] = nodos_duraciones.get(nombre_nodo, 0) + dur
            elif tipo in ["DECISION", "GATEWAY"]:
                decisiones_count += 1
                dur = random.randint(5, 30)
                duracion_total += dur
                nodos_duraciones[nombre_nodo] = nodos_duraciones.get(nombre_nodo, 0) + dur
            elif tipo == "FORK":
                forks += 1
            elif tipo == "JOIN":
                joins += 1
                
            opciones = edges[current]
            
            # Simular decisiones complejas
            if len(opciones) > 1 or tipo == "DECISION":
                is_happy = random.random() < 0.75
                if is_happy:
                    next_trans = opciones[0]
                    decisiones.append(f"{nombre_nodo} aprobado=SI")
                else:
                    next_trans = random.choice(opciones)
                    decisiones.append(f"{nombre_nodo} aprobado=NO|Requiere corrección=SI")
            else:
                next_trans = opciones[0]
                
            next_node = str(next_trans.get("destino"))
            
            if next_node in path_ids:
                retornos_count += 1
                
            path_ids.append(next_node)
            if next_node in nodos_dict:
                path.append(nodos_dict[next_node].get("nombre", next_node))
                
            current = next_node
            nodos_count += 1
            
        actividad_lenta = max(nodos_duraciones, key=nodos_duraciones.get) if nodos_duraciones else "Ninguna"
        carril_lento = carriles_visitados[-1] if carriles_visitados else "General"
        cantidad_documentos = random.randint(1, 5)
        
        # LÓGICA DE ANOMALÍAS EXPANDIDA
        es_anomalo = False
        # 1. Tiempos excesivos (> 1000 mins)
        if duracion_total > 1000:
            es_anomalo = True
        # 2. Demasiadas tareas repetidas o retornos
        if retornos_count >= 2:
            es_anomalo = True
        # 3. Trámite se desvía del comportamiento esperado (demasiadas decisiones sin fin)
        if decisiones_count >= 3 and random.random() < 0.4:
            es_anomalo = True
        # 4. Falta de documentos donde se requiere
        if nodos_count > 4 and cantidad_documentos == 0 and random.random() < 0.3:
            es_anomalo = True
        # 5. Forzar anomalías aleatorias (25% de los casos) para balancear el dataset
        if random.random() < 0.25:
            es_anomalo = True
            
        anomalia = 1 if es_anomalo else 0
        riesgo_demora = 1 if (duracion_total > 500 or retornos_count > 0) else 0
        cuello_botella = 1 if (duracion_total > 600 or random.random() < 0.2) else 0
        
        prioridad_actual = random.choices(["BAJA", "NORMAL", "ALTA"], weights=[0.3, 0.5, 0.2])[0]
        prioridad_rec = "ALTA" if riesgo_demora == 1 else ("NORMAL" if prioridad_actual == "ALTA" else "BAJA")
        
        ruta_str = "->".join(path_ids)
        ruta_legible = " -> ".join(path)
        
        happy_path = [p for p in path_ids if path_ids.count(p) == 1]
        
        motivo = "Ruta directa recomendada porque reduce tiempos y retornos."
        if cuello_botella:
            motivo += f" Posible cuello de botella en {actividad_lenta}."
        if riesgo_demora:
            motivo += " Riesgo alto por múltiples observaciones."

        escenario = {
            "instanciaId": f"TRN-{uuid.uuid4().hex[:6].upper()}",
            "politicaId": politica.get("id", "UNKNOWN"),
            "nombrePolitica": politica.get("nombre", "UNKNOWN"),
            "origenDatos": "ENTRENAMIENTO_REAL",
            "rutaEjecutadaCodificada": ruta_str,
            "rutaEjecutadaLegible": ruta_legible,
            "carrilesVisitados": "->".join(carriles_visitados) if carriles_visitados else "General",
            "actividadesVisitadas": "->".join(visited_activities) if visited_activities else "Ninguna",
            "decisionesTomadas": "|".join(decisiones) if decisiones else "Flujo sin desviaciones",
            "cantidadObservaciones": random.randint(1, 3) if retornos_count > 0 else 0,
            "cantidadNodos": nodos_count,
            "cantidadDecisiones": decisiones_count,
            "cantidadForks": forks,
            "cantidadJoins": joins,
            "cantidadRetornos": retornos_count,
            "cantidadReprocesos": retornos_count,
            "cantidadDocumentos": cantidad_documentos,
            "cantidadFuncionariosInvolucrados": len(carriles_visitados) + random.randint(0, 2),
            "duracionPromedioHistorica": float(duracion_total),
            "nodoMasLento": actividad_lenta,
            "actividadMasLenta": actividad_lenta,
            "carrilMasLento": carril_lento,
            "cuelloBotellaLabel": cuello_botella,
            "anomaliaLabel": anomalia,
            "riesgoDemoraLabel": riesgo_demora,
            "prioridadActual": prioridad_actual,
            "prioridadRecomendadaLabel": prioridad_rec,
            "rutaRecomendadaLabel": "RUTA_OPTIMA",
            "rutaRecomendadaLegible": " -> ".join([nodos_dict.get(n, {}).get("nombre", n) for n in happy_path]) if happy_path else ruta_legible,
            "motivoRecomendacion": motivo
        }
        
        escenarios.append(escenario)
        
    return escenarios

async def main():
    politicas = await get_politicas_reales()
    todos = []
    print(f"Generando 2,000 datos altamente realistas para {len(politicas)} politicas...")
    for pol in politicas:
        print(f"-> Procesando politica: {pol.get('nombre')}")
        escenarios = generar_dataset_avanzado(pol, 25000)
        todos.extend(escenarios)
        
    print(f"Total datos generados: {len(todos)}. Guardando en CSV...")
    csv_path, json_path = save_synthetic_dataset(todos)
    print(f"Guardado en: {csv_path}")

if __name__ == "__main__":
    asyncio.run(main())
