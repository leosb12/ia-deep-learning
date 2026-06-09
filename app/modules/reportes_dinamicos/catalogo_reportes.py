"""
Catálogo de entidades, campos y operaciones permitidas.
El Motor IA solo puede generar planes que referencien elementos de este catálogo.
El backend valida contra este catálogo antes de ejecutar cualquier consulta.
"""
import os
import requests
from typing import Dict, List, Set



# ============================================================
# CATÁLOGO DE ENTIDADES PERMITIDAS (MongoDB)
# ============================================================
CATALOGO_MONGO: Dict[str, List[str]] = {
    "instancias_politica": [
        "id", "codigoTramite", "estadoInstancia", "fechaCreacion", "fechaFinalizacion",
        "creadaPor", "departamentoId", "departamentoActual", "politicaId", "politicaNombre",
        "funcionarioAsignado", "requierePago", "prioridad", "estado"
    ],
    "politicas_negocio": [
        "id", "nombre", "categoria", "estado", "requierePago", "version",
        "fechaCreacion", "descripcion", "activo"
    ],
    "usuarios": [
        "id", "nombre", "correo", "rol", "departamentoId", "activo",
        "fechaRegistro", "telefono"
    ],
    "pagos": [
        "id", "instanciaPoliticaId", "politicaId", "monto", "estado",
        "fechaCreacion", "metodoPago", "referencia"
    ],
    "tareas_actividad": [
        "id", "instanciaId", "responsableId", "estado", "fechaCreacion",
        "fechaLimite", "fechaCompletado", "actividadNombre", "tipo"
    ],
    "departamentos": [
        "id", "nombre", "responsableId", "descripcion", "activo"
    ],
    "notificaciones": [
        "id", "usuarioId", "tipo", "mensaje", "leida", "fechaCreacion"
    ],
    "formularios_dinamicos": [
        "id", "instanciaId", "politicaId", "datos", "fechaCreacion"
    ],
    "requisitos_iniciales": [
        "id", "politicaId", "nombre", "tipo", "obligatorio"
    ],
    "trazabilidad": [
        "id", "instanciaId", "accion", "usuarioId", "fechaAccion",
        "detalle", "nodoId", "nodoNombre"
    ],
    "predicciones_ia": [
        "id", "instanciaId", "politicaId", "cuelloBotella", "riesgoDemora",
        "prioridadRecomendada", "rutaRecomendada", "fechaPrediccion"
    ],
}

# ============================================================
# CATÁLOGO DE FUENTES DYNAMO (metadatos de archivos)
# ============================================================
CATALOGO_DYNAMO: Dict[str, List[str]] = {
    "metadatos_archivos": [
        "archivoId", "tramiteId", "politicaId", "usuarioSubio", "fechaSubida",
        "nombreArchivo", "tipoArchivo", "extension", "tamano",
        "estadoProcesamiento", "url"
    ],
}

# ============================================================
# OPERACIONES PERMITIDAS
# ============================================================
OPERACIONES_PERMITIDAS: Set[str] = {
    "count", "sum", "avg", "max", "min", "ranking",
    "listado", "detalle", "resumen", "distribucion", "tendencia"
}

# ============================================================
# OPERADORES DE FILTRO PERMITIDOS
# ============================================================
OPERADORES_PERMITIDOS: Set[str] = {
    "=", "!=", ">", ">=", "<", "<=",
    "mes_actual", "anio_actual", "ultimos_dias", "ultimos_meses",
    "contiene", "no_contiene", "entre", "en_lista"
}

# ============================================================
# TIPOS DE CONSULTA PERMITIDOS
# ============================================================
TIPOS_CONSULTA: Set[str] = {
    "analitica", "detalle", "ranking", "distribucion",
    "tendencia", "resumen", "listado", "busqueda"
}

# ============================================================
# CATÁLOGO REMOTO
# ============================================================
BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://localhost:8080").rstrip("/")
BACKEND_REPORTES_CATALOGO_URL = os.getenv("BACKEND_REPORTES_CATALOGO_URL", f"{BACKEND_BASE_URL}/api/admin/reportes/catalogo/resumido")


def get_catalogo_remoto():
    try:
        response = requests.get(BACKEND_REPORTES_CATALOGO_URL, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Error al obtener catálogo de Spring Boot: {e}")
    return None

def build_catalogo_from_remote(remote_data):
    if not remote_data or "entidades" not in remote_data:
        return CATALOGO_MONGO, OPERADORES_PERMITIDOS, OPERACIONES_PERMITIDAS, "FALLBACK_LOCAL"
    
    catalogo = {}
    for ent in remote_data["entidades"]:
        campos = [c["nombreLogico"] for c in ent.get("campos", [])]
        catalogo[ent["nombreLogico"]] = campos
    
    operadores = set(remote_data.get("operadoresPermitidos", list(OPERADORES_PERMITIDOS)))
    operaciones = set(remote_data.get("operacionesPermitidas", list(OPERACIONES_PERMITIDAS)))
    return catalogo, operadores, operaciones, "SPRING_BOOT"

def get_catalogo_completo() -> dict:
    """Retorna el catálogo completo de entidades y campos permitidos."""
    remote_data = get_catalogo_remoto()
    cat_mongo, operadores, operaciones, origen = build_catalogo_from_remote(remote_data)
    
    return {
        "origen": origen,
        "mongo": cat_mongo,
        "dynamo": CATALOGO_DYNAMO,
        "operaciones": list(operaciones),
        "operadores": list(operadores),
        "tiposConsulta": list(TIPOS_CONSULTA),
    }

def es_entidad_permitida(entidad: str) -> bool:
    """Verifica si una entidad está en el catálogo seguro."""
    if not entidad:
        return False
    cat = get_catalogo_completo()
    return entidad in cat["mongo"] or entidad in cat["dynamo"]

def es_campo_permitido(entidad: str, campo: str) -> bool:
    """Verifica si un campo está permitido para una entidad dada."""
    if not entidad or not campo:
        return True  # Permitir campos vacíos para count(*)
    
    cat = get_catalogo_completo()
    campos = cat["mongo"].get(entidad, []) or cat["dynamo"].get(entidad, [])
    return campo in campos or campo in ("id", "_id")

def es_operacion_permitida(operacion: str) -> bool:
    """Verifica si una operación es permitida."""
    cat = get_catalogo_completo()
    return operacion.lower() in cat["operaciones"] if operacion else True


def validar_plan_consulta(plan: dict) -> List[str]:
    """
    Valida un plan de consulta generado por la IA contra el catálogo.
    Retorna una lista de errores encontrados. Lista vacía = plan válido.
    """
    errores = []
    
    entidad = plan.get("entidadPrincipal", "")
    if entidad and not es_entidad_permitida(entidad):
        errores.append(f"Entidad no permitida: {entidad}")
    
    # Validar campos
    for campo in plan.get("camposSolicitados", []):
        if not es_campo_permitido(entidad, campo):
            errores.append(f"Campo no permitido: {campo} en {entidad}")
    
    # Validar filtros
    cat = get_catalogo_completo()
    for filtro in plan.get("filtros", []):
        campo_f = filtro.get("campo", "")
        if campo_f and not es_campo_permitido(entidad, campo_f):
            errores.append(f"Campo de filtro no permitido: {campo_f}")
        operador = filtro.get("operador", "")
        if operador and operador not in cat["operadores"]:
            errores.append(f"Operador no permitido: {operador}")
    
    # Validar agrupaciones
    for grupo in plan.get("agrupaciones", []):
        if not es_campo_permitido(entidad, grupo):
            errores.append(f"Campo de agrupación no permitido: {grupo}")
    
    # Validar fuentes
    for fuente in plan.get("fuentesNecesarias", []):
        partes = fuente.split(".")
        if len(partes) == 2:
            tipo, ent = partes
            if tipo == "mongo" and ent not in cat["mongo"]:
                errores.append(f"Fuente Mongo no permitida: {ent}")
            elif tipo == "dynamo" and ent not in cat["dynamo"]:
                errores.append(f"Fuente Dynamo no permitida: {ent}")
    
    return errores


def get_contexto_catalogo_para_ia() -> str:
    """
    Genera una representación textual del catálogo para incluir en prompts de IA.
    La IA usa esto para saber qué entidades y campos puede referenciar.
    """
    cat = get_catalogo_completo()
    lineas = ["CATÁLOGO DE DATOS DISPONIBLES:"]
    lineas.append(f"ORIGEN DEL CATÁLOGO: {cat['origen']}")
    lineas.append("\n== MongoDB ==")
    for entidad, campos in cat["mongo"].items():
        lineas.append(f"  • {entidad}: {', '.join(campos)}")
    
    lineas.append("\n== DynamoDB (metadatos archivos) ==")
    for entidad, campos in cat["dynamo"].items():
        lineas.append(f"  • {entidad}: {', '.join(campos)}")
    
    lineas.append(f"\n== Operaciones permitidas ==: {', '.join(cat['operaciones'])}")
    lineas.append(f"== Operadores de filtro ==: {', '.join(cat['operadores'])}")
    
    return "\n".join(lineas)
