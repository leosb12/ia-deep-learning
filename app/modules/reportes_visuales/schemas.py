from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class ReporteVisualRequest(BaseModel):
    prompt: str
    usuarioId: Optional[str] = None
    iaPlus: Optional[bool] = False
    usuariosReales: Optional[List[str]] = None
    politicasReales: Optional[List[str]] = None

class ResultadoBloqueDatos(BaseModel):
    labels: Optional[List[str]] = []
    values: Optional[List[float]] = []
    columns: Optional[List[str]] = []
    rows: Optional[List[List[Any]]] = []

class BloqueReporteIntent(BaseModel):
    tipo: str  # bar, pie, doughnut, line, area, table, matrix, kpi, error
    intencion: str
    titulo: str
    orden: int
    entidadPrincipal: str
    metrica: str
    limite: int = 10
    filtros: Optional[Dict[str, Any]] = None
    datos: Optional[ResultadoBloqueDatos] = None

class ReporteVisualResponse(BaseModel):
    titulo: str
    descripcion: str
    bloques: List[BloqueReporteIntent]
