"""
Esquemas Pydantic para el módulo de Reportes Inteligentes y Asistente de Datos.
Define los contratos de entrada/salida para todos los endpoints del motor IA.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


# ============================================================
# ENUMS
# ============================================================

class MotorTipo(str, Enum):
    MOTOR_INTERNO = "MOTOR_INTERNO"
    MOTOR_IA_AVANZADO = "MOTOR_IA_AVANZADO"
    MOTOR_FALLBACK = "MOTOR_FALLBACK"


class TipoConsulta(str, Enum):
    ANALITICA = "analitica"
    DETALLE = "detalle"
    RANKING = "ranking"
    DISTRIBUCION = "distribucion"
    TENDENCIA = "tendencia"
    RESUMEN = "resumen"
    LISTADO = "listado"
    BUSQUEDA = "busqueda"


# ============================================================
# SCHEMAS COMUNES
# ============================================================

class Metrica(BaseModel):
    operacion: str
    campo: str
    alias: str


class Filtro(BaseModel):
    campo: str
    operador: str
    valor: Optional[Any] = None


class Ordenamiento(BaseModel):
    campo: str
    direccion: str = "desc"


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class ReporteRequest(BaseModel):
    texto: str
    usuarioId: str = ""
    rol: str = "ADMIN"


class AsistenteDatosRequest(BaseModel):
    texto: str
    usuarioId: str = ""
    rol: str = "ADMIN"
    contextoAdicional: Optional[str] = None


class PlanEjecucionRequest(BaseModel):
    """Request para ejecutar un plan ya generado y validado."""
    plan: Dict[str, Any]
    textoOriginal: str = ""
    usuarioId: str = ""
    rol: str = "ADMIN"


# ============================================================
# RESPONSE SCHEMAS - REPORTES
# ============================================================

class ReporteResponse(BaseModel):
    titulo: str = "Reporte dinámico"
    descripcion: str = "Descripción del reporte solicitado"
    intencionDetectada: str = "ambiguo"
    entidadPrincipal: Optional[str] = None
    campos: List[str] = []
    metricas: List[Metrica] = []
    filtros: List[Filtro] = []
    agrupaciones: List[str] = []
    ordenamiento: List[Ordenamiento] = []
    limite: int = 50
    formatoSalida: str = "pantalla"
    visualizacion: str = "tabla"
    requiereAclaracion: bool = False
    preguntaAclaratoria: Optional[str] = None
    opcionesSugeridas: List[str] = []
    confianza: float = 0.0
    motor: str = MotorTipo.MOTOR_FALLBACK
    respuestaNatural: Optional[str] = None
    requiereResolucionBackend: Optional[bool] = None


# ============================================================
# RESPONSE SCHEMAS - ASISTENTE DE DATOS
# ============================================================

class PlanConsulta(BaseModel):
    """Plan de consulta generado por la IA para ser validado por el backend."""
    requiereDatos: bool = True
    tipoConsulta: str = "analitica"
    fuentesNecesarias: List[str] = []
    entidadPrincipal: str = ""
    operacion: str = "listado"
    camposSolicitados: List[str] = []
    filtros: List[Filtro] = []
    agrupaciones: List[str] = []
    ordenamiento: List[Ordenamiento] = []
    limite: int = 50
    requiereBusquedaSemantica: bool = False
    requiereAclaracion: bool = False
    preguntaAclaratoria: Optional[str] = None


class AsistenteDatosResponse(BaseModel):
    """Respuesta final del asistente de datos."""
    respuesta: str = ""
    resumen: str = ""
    datos: List[Dict[str, Any]] = []
    columnas: List[str] = []
    visualizacionSugerida: str = "tabla"
    accionesSugeridas: List[str] = []
    fuentesConsultadas: List[str] = []
    advertencias: List[str] = []
    plan: Optional[PlanConsulta] = None
    motor: str = MotorTipo.MOTOR_FALLBACK
    confianza: float = 0.0


class TranscripcionRequest(BaseModel):
    audio_base64: str


class CatalogoResponse(BaseModel):
    mongo: Dict[str, List[str]] = {}
    dynamo: Dict[str, List[str]] = {}
    operaciones: List[str] = []
    operadores: List[str] = []
    tiposConsulta: List[str] = []


class AsistenciaExtendidaRequest(BaseModel):
    preguntaOriginal: str
    columnasEsperadas: List[str]
    diagnosticoMotorReal: Optional[str] = None
    datosRealesDisponibles: Optional[List[Dict[str, Any]]] = None
    datosSimuladosPrevios: Optional[List[Dict[str, Any]]] = None
    usuariosReales: Optional[List[str]] = None
    funcionariosReales: Optional[List[str]] = None
    administradoresReales: Optional[List[str]] = None
    politicasReales: Optional[List[str]] = None
    departamentosReales: Optional[List[str]] = None
    estadosReales: Optional[List[str]] = None
    nombresNodosReales: Optional[List[str]] = None


class AsistenciaExtendidaResponse(BaseModel):
    columnas: List[str]
    filas: List[Dict[str, Any]]
    asistido: bool = True

