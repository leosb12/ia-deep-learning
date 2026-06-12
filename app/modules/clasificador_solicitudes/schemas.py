from pydantic import BaseModel, Field, field_validator


class SolicitudClasificacion(BaseModel):
    texto: str = Field(..., min_length=1, examples=["mi internet esta muy lento y se corta"])

    @field_validator("texto")
    @classmethod
    def validar_texto_no_vacio(cls, value: str) -> str:
        return value.strip()


class RequisitoInicial(BaseModel):
    nombre: str
    label: str | None = None
    tipo: str | None = None
    obligatorio: bool = False


class ResultadoPolitica(BaseModel):
    politicaId: str
    nombrePolitica: str | None = None
    confianza: float
    scoreRequisitos: float | None = None
    scoreSemantico: float | None = None
    scoreFinal: float | None = None
    requisitosCoincidentes: list[str] = Field(default_factory=list)
    requisitosFaltantes: list[str] = Field(default_factory=list)


class DatoExtraido(BaseModel):
    campo: str = ""
    valor: str = ""
    textoOriginal: str = ""


class RequisitoDetectado(BaseModel):
    nombre: str = ""
    valor: str = ""
    politicaId: str | None = None


class CampoFaltante(BaseModel):
    nombre: str = ""
    motivo: str = ""


class AnalisisDeepSeek(BaseModel):
    intencionPrincipal: str | None = None
    politicaSugeridaId: str | None = None
    politicaSugeridaNombre: str | None = None
    confianza: float = 0.0
    informacionEntregada: list[DatoExtraido] = Field(default_factory=list)
    requisitosDetectados: list[RequisitoDetectado] = Field(default_factory=list)
    camposFaltantes: list[CampoFaltante] = Field(default_factory=list)
    observaciones: list[str] = Field(default_factory=list)


class RespuestaClasificacion(BaseModel):
    politicaId: str
    nombrePolitica: str | None = None
    confianza: float
    origen: str
    requiereMasInformacion: bool = False
    topResultados: list[ResultadoPolitica]


class RespuestaClasificacionDinamica(RespuestaClasificacion):
    metodoRecomendacion: str | None = None
    requisitosDetectados: list[str] = Field(default_factory=list)
    requisitosCoincidentes: list[str] = Field(default_factory=list)
    requisitosFaltantes: list[str] = Field(default_factory=list)
    analisisDeepSeek: AnalisisDeepSeek | None = None


class PoliticaDinamica(BaseModel):
    id: str = Field(..., min_length=1)
    nombre: str = Field(..., min_length=1)
    descripcion: str | None = None
    categoria: str | None = None
    descripcionClasificacion: str | None = None
    palabrasClave: list[str] = Field(default_factory=list)
    intencionesEjemplo: list[str] = Field(default_factory=list)
    requisitosSugeridos: list[str] = Field(default_factory=list)
    requisitosIniciales: list[RequisitoInicial] = Field(default_factory=list)

    @field_validator("id", "nombre")
    @classmethod
    def validar_texto_obligatorio(cls, value: str) -> str:
        return value.strip()

    @field_validator("palabrasClave", "intencionesEjemplo", "requisitosSugeridos", mode="before")
    @classmethod
    def validar_listas(cls, value):
        if value is None:
            return []
        return [str(item).strip() for item in value if item and str(item).strip()]


class SolicitudClasificacionDinamica(BaseModel):
    texto: str = Field(..., min_length=1)
    politicas: list[PoliticaDinamica] = Field(..., min_length=1)
    usarDeepSeek: bool = False
    nombreDocumento: str | None = None
    usarSoloRequisitosIniciales: bool = False

    @field_validator("texto")
    @classmethod
    def validar_texto_no_vacio(cls, value: str) -> str:
        return value.strip()
