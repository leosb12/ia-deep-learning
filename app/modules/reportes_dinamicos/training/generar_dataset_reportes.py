import json
import random
import csv
import os

# Generador de dataset masivo para intenciones de reportes (100,000+ ejemplos)

INTENCIONES_BASE = [
    {
        "intencion": "ranking_politicas_mas_utilizadas",
        "entidadPrincipal": "instancias_politica",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "cantidadTramites"}],
        "agrupaciones": ["politicaNombre"],
        "ordenamiento": [{"campo": "cantidadTramites", "direccion": "desc"}],
        "patrones": [
            "quiero ver la politica mas usada {tiempo}",
            "cuales son las politicas mas usadas {tiempo}",
            "dame las politicas mas utilizadas {tiempo}",
            "muestrame la politica q mas se usa {tiempo}",
            "ranking de politicas usadas {tiempo}",
            "que politica se uso mas {tiempo}"
        ],
        "default_visualizacion": "grafico_barras"
    },
    {
        "intencion": "ranking_clientes_por_tramites",
        "entidadPrincipal": "instancias_politica",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "cantidadTramites"}],
        "agrupaciones": ["creadaPor"],
        "ordenamiento": [{"campo": "cantidadTramites", "direccion": "desc"}],
        "patrones": [
            "muestrame los clientes que mas tramites iniciaron {tiempo}",
            "quiero ver q cliente tiene mas tramites {tiempo}",
            "clientes con mas tramites {tiempo}",
            "ranking de clientes por cantidad de tramites {tiempo}",
            "dame los usuarios q iniciaron mas tramites {tiempo}"
        ],
        "default_visualizacion": "tabla"
    },
    {
        "intencion": "tramites_por_estado_departamento",
        "entidadPrincipal": "instancias_politica",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "cantidadTramites"}],
        "agrupaciones": ["estadoInstancia", "departamentoId"], # Asumiendo departamento del estado actual
        "ordenamiento": [{"campo": "cantidadTramites", "direccion": "desc"}],
        "patrones": [
            "genera un reporte de tramites {estado} agrupados por departamento",
            "quiero ver tramites {estado} por departamento",
            "dame los tramites {estado} separados por depto",
            "muestrame los tramites {estado} y agrupalos por departamento",
            "reporte de tramites {estado} x departamento"
        ],
        "default_visualizacion": "grafico_pie"
    },
    {
        "intencion": "pagos_por_politica",
        "entidadPrincipal": "pagos",
        "metricas": [{"operacion": "sum", "campo": "monto", "alias": "totalPagos"}],
        "agrupaciones": ["politicaId"],
        "ordenamiento": [{"campo": "totalPagos", "direccion": "desc"}],
        "patrones": [
            "agrupa los pagos por politica y ordenalos de mayor a menor",
            "dame los pagos por politica de mayor a menor",
            "quiero ver cuanto se pago por cada politica",
            "muestrame el total de pagos agrupado por politica",
            "reporte de ingresos x politica"
        ],
        "default_visualizacion": "tabla"
    },
    {
        "intencion": "tareas_pendientes_funcionario",
        "entidadPrincipal": "tareas_actividad",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "tareasPendientes"}],
        "agrupaciones": ["responsableId"],
        "ordenamiento": [{"campo": "tareasPendientes", "direccion": "desc"}],
        "patrones": [
            "muestrame que funcionario tiene mas tareas pendientes",
            "quiero saber quien tiene mas tareas",
            "ranking de funcionarios con tareas pendientes",
            "dame las tareas pendientes por usuario",
            "quien tiene mas trabajo acumulado"
        ],
        "default_visualizacion": "tabla"
    }
]

AMBIGUOS = [
    {
        "texto": "Quiero un reporte de clientes.",
        "preguntaAclaratoria": "¿Qué quieres analizar de los clientes: cantidad de trámites iniciados, pagos realizados, trámites pendientes o trámites finalizados?"
    },
    {
        "texto": "Hazme un reporte de trámites.",
        "preguntaAclaratoria": "¿Quieres ver los trámites agrupados por estado, por cliente, por departamento o por política?"
    },
    {
        "texto": "Muéstrame las políticas.",
        "preguntaAclaratoria": "¿Deseas ver las políticas más usadas, las que generan más ingresos o un listado general?"
    },
    {
        "texto": "Quiero ver pagos.",
        "preguntaAclaratoria": "¿Te gustaría ver los pagos agrupados por política, por cliente, por fecha o por estado?"
    },
    {
        "texto": "Necesito estadísticas.",
        "preguntaAclaratoria": "¿Sobre qué entidad deseas las estadísticas? (Políticas, Trámites, Clientes, Pagos)"
    }
]

TIEMPOS = {
    "este mes": {"campo": "fechaCreacion", "operador": "mes_actual", "valor": None},
    "este año": {"campo": "fechaCreacion", "operador": "anio_actual", "valor": None},
    "ultimos 7 dias": {"campo": "fechaCreacion", "operador": "ultimos_dias", "valor": 7},
    "ultimos 30 dias": {"campo": "fechaCreacion", "operador": "ultimos_dias", "valor": 30},
    "ultimos 3 meses": {"campo": "fechaCreacion", "operador": "ultimos_meses", "valor": 3},
    "entre enero y marzo": {"campo": "fechaCreacion", "operador": "rango_fechas", "valor": "01-01|03-31"},
    "": None
}

ESTADOS = {
    "pendientes": {"campo": "estadoInstancia", "operador": "=", "valor": "EN_CURSO"}, # EN_PROGRESO o EN_CURSO
    "finalizados": {"campo": "estadoInstancia", "operador": "=", "valor": "COMPLETADO"},
    "rechazados": {"campo": "estadoInstancia", "operador": "=", "valor": "RECHAZADO"},
    "cancelados": {"campo": "estadoInstancia", "operador": "=", "valor": "CANCELADO"},
    "": None
}

FORMATOS = {
    "en excel": "excel",
    "en exel": "excel",
    "en pdf": "pdf",
    "en word": "word",
    "en pantalla": "pantalla",
    "para descargar": "excel",
    "exportalo en pdf": "pdf",
    "": "pantalla"
}

def generate_dataset(num_samples=100000):
    dataset = []
    
    # 1. Ejemplos completos y variantes
    while len(dataset) < num_samples - len(AMBIGUOS)*1000:
        base = random.choice(INTENCIONES_BASE)
        patron = random.choice(base["patrones"])
        
        tiempo_k = random.choice(list(TIEMPOS.keys()))
        estado_k = random.choice(list(ESTADOS.keys()))
        formato_k = random.choice(list(FORMATOS.keys()))
        
        texto = patron.replace("{tiempo}", tiempo_k).replace("{estado}", estado_k)
        if formato_k:
            texto += " " + formato_k
            
        texto = texto.strip().replace("  ", " ")
        
        filtros = []
        if tiempo_k and TIEMPOS[tiempo_k]:
            filtros.append(TIEMPOS[tiempo_k])
        if "{estado}" in patron and estado_k and ESTADOS[estado_k]:
            filtros.append(ESTADOS[estado_k])
            
        formato = FORMATOS[formato_k]
        
        # Ocasionalmente añadir errores tipograficos
        if random.random() < 0.1:
            texto = texto.replace("tramites", "tramietes").replace("politicas", "pliticas").replace("mayor", "mayr")
            
        json_obj = {
            "intencionDetectada": base["intencion"],
            "entidadPrincipal": base["entidadPrincipal"],
            "campos": [],
            "metricas": base["metricas"],
            "filtros": filtros,
            "agrupaciones": base["agrupaciones"],
            "ordenamiento": base["ordenamiento"],
            "limite": 100,
            "formatoSalida": formato,
            "visualizacion": base["default_visualizacion"],
            "requiereAclaracion": False,
            "preguntaAclaratoria": None
        }
        
        dataset.append({
            "texto_usuario": texto.lower(),
            "intencion": base["intencion"],
            "entidadPrincipal": base["entidadPrincipal"],
            "formatoSalida": formato,
            "visualizacion": base["default_visualizacion"],
            "requiereAclaracion": "False",
            "json_objetivo": json.dumps(json_obj)
        })

    # 2. Ejemplos ambiguos
    for _ in range(len(AMBIGUOS)*1000):
        amb = random.choice(AMBIGUOS)
        texto = amb["texto"]
        if random.random() < 0.3:
            texto = texto.lower()
            
        json_obj = {
            "requiereAclaracion": True,
            "preguntaAclaratoria": amb["preguntaAclaratoria"],
            "intencionDetectada": "ambiguo",
            "entidadPrincipal": None
        }
        
        dataset.append({
            "texto_usuario": texto,
            "intencion": "ambiguo",
            "entidadPrincipal": "",
            "formatoSalida": "pantalla",
            "visualizacion": "",
            "requiereAclaracion": "True",
            "json_objetivo": json.dumps(json_obj)
        })
        
    random.shuffle(dataset)
    return dataset[:num_samples]

if __name__ == "__main__":
    from pathlib import Path
    print("Generando dataset...")
    data = generate_dataset(100000)
    
    base_dir = Path(__file__).resolve().parent.parent
    datasets_dir = base_dir / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = datasets_dir / "dataset_reportes.csv"
    
    with open(dataset_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["texto_usuario", "intencion", "entidadPrincipal", "formatoSalida", "visualizacion", "requiereAclaracion", "json_objetivo"])
        writer.writeheader()
        writer.writerows(data)
    print("Dataset generado en", dataset_path, "con", len(data), "registros.")
