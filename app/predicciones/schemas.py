from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict

class DatasetRequest(BaseModel):
    cantidadPorPolitica: int = 100
    usarPoliticasReales: bool = True
    incluirCasosConDemora: bool = True
    incluirCasosConCorreccion: bool = True
    incluirCasosConPago: bool = True
    incluirAnomalias: bool = True

class PredictionRequest(BaseModel):
    politicaId: str
    nombrePolitica: str
    cantidadObservaciones: int = 0
    cantidadNodos: int = 0
    cantidadDecisiones: int = 0
    cantidadForks: int = 0
    cantidadJoins: int = 0
    cantidadRetornos: int = 0
    cantidadReprocesos: int = 0
    cantidadDocumentos: int = 0
    cantidadFuncionariosInvolucrados: int = 0
    duracionPromedioHistorica: float = 0.0
    prioridadActual: str = "NORMAL"
    rutaEjecutadaCodificada: str = ""
    rutaEjecutadaLegible: str = ""
    carrilesVisitados: str = ""
    actividadesVisitadas: str = ""
    politicaEstructuraJson: Optional[str] = None

class PredictionResponse(BaseModel):
    riesgoDemora: str
    probabilidadRiesgoDemora: float
    cuelloBotella: str
    probabilidadCuelloBotella: float
    anomalia: str
    probabilidadAnomalia: float
    prioridadRecomendada: str
    rutaRecomendadaLabel: str
    rutaRecomendadaLegible: str
    confianzaRuta: float
    mensaje: str

class ExplainRequest(BaseModel):
    prediccion: PredictionResponse

class SyntheticScenario(BaseModel):
    instanciaId: str = Field(default="")
    politicaId: str = Field(default="")
    nombrePolitica: str = Field(default="")
    origenDatos: str = Field(default="SIMULADO_DEEPSEEK")
    rutaEjecutadaCodificada: str = Field(default="")
    rutaEjecutadaLegible: str = Field(default="")
    carrilesVisitados: str = Field(default="")
    actividadesVisitadas: str = Field(default="")
    decisionesTomadas: str = Field(default="")
    cantidadObservaciones: int = Field(default=0)
    cantidadNodos: int = Field(default=0)
    cantidadDecisiones: int = Field(default=0)
    cantidadForks: int = Field(default=0)
    cantidadJoins: int = Field(default=0)
    cantidadRetornos: int = Field(default=0)
    cantidadReprocesos: int = Field(default=0)
    cantidadDocumentos: int = Field(default=0)
    cantidadFuncionariosInvolucrados: int = Field(default=0)
    duracionPromedioHistorica: float = Field(default=0.0)
    nodoMasLento: str = Field(default="")
    actividadMasLenta: str = Field(default="")
    carrilMasLento: str = Field(default="")
    cuelloBotellaLabel: int = Field(default=0)
    anomaliaLabel: int = Field(default=0)
    riesgoDemoraLabel: int = Field(default=0)
    prioridadActual: str = Field(default="NORMAL")
    prioridadRecomendadaLabel: str = Field(default="NORMAL")
    rutaRecomendadaLabel: str = Field(default="")
    rutaRecomendadaLegible: str = Field(default="")
    motivoRecomendacion: str = Field(default="")

class DeepSeekResponse(BaseModel):
    escenarios: List[SyntheticScenario]
