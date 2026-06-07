from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import os
import re
from pathlib import Path

router = APIRouter(prefix="/api/ia/reportes", tags=["Reportes Inteligentes"])

class ReporteRequest(BaseModel):
    texto: str
    usuarioId: str
    rol: str

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
    direccion: str

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
    limite: int = 100
    formatoSalida: str = "pantalla"
    visualizacion: str = "tabla"
    requiereAclaracion: bool = False
    preguntaAclaratoria: Optional[str] = None
    confianza: float = 0.0

class TranscripcionRequest(BaseModel):
    # This would take audio file bytes, but keeping it simple for now based on instructions
    audio_base64: str

# Paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "modelo_reportes.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
ENCODERS_PATH = MODELS_DIR / "label_encoders.pkl"

model = None
tokenizer = None
le_intent = None
le_format = None

def load_resources():
    global model, tokenizer, le_intent, le_format
    if model is None and MODEL_PATH.exists():
        model = load_model(str(MODEL_PATH))
        with open(TOKENIZER_PATH, 'rb') as handle:
            tokenizer = pickle.load(handle)
        with open(ENCODERS_PATH, 'rb') as handle:
            encoders = pickle.load(handle)
            le_intent = encoders['intent']
            le_format = encoders['format']

# Dictionaries representing the business logic mappings
INTENT_MAPPINGS = {
    "ranking_politicas_mas_utilizadas": {
        "entidadPrincipal": "instancias_politica",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "cantidadTramites"}],
        "agrupaciones": ["politicaNombre"],
        "ordenamiento": [{"campo": "cantidadTramites", "direccion": "desc"}],
        "visualizacion": "grafico_barras"
    },
    "ranking_clientes_por_tramites": {
        "entidadPrincipal": "instancias_politica",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "cantidadTramites"}],
        "agrupaciones": ["creadaPor"],
        "ordenamiento": [{"campo": "cantidadTramites", "direccion": "desc"}],
        "visualizacion": "tabla"
    },
    "tramites_por_estado_departamento": {
        "entidadPrincipal": "instancias_politica",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "cantidadTramites"}],
        "agrupaciones": ["estadoInstancia", "departamentoId"],
        "ordenamiento": [{"campo": "cantidadTramites", "direccion": "desc"}],
        "visualizacion": "grafico_pie"
    },
    "pagos_por_politica": {
        "entidadPrincipal": "pagos",
        "metricas": [{"operacion": "sum", "campo": "monto", "alias": "totalPagos"}],
        "agrupaciones": ["politicaId"],
        "ordenamiento": [{"campo": "totalPagos", "direccion": "desc"}],
        "visualizacion": "tabla"
    },
    "tareas_pendientes_funcionario": {
        "entidadPrincipal": "tareas_actividad",
        "metricas": [{"operacion": "count", "campo": "id", "alias": "tareasPendientes"}],
        "agrupaciones": ["responsableId"],
        "ordenamiento": [{"campo": "tareasPendientes", "direccion": "desc"}],
        "visualizacion": "tabla"
    }
}

def extract_filters(texto: str) -> List[dict]:
    filtros = []
    texto = texto.lower()
    
    # Tiempos
    if "este mes" in texto:
        filtros.append({"campo": "fechaCreacion", "operador": "mes_actual", "valor": None})
    elif "este año" in texto or "este ano" in texto:
        filtros.append({"campo": "fechaCreacion", "operador": "anio_actual", "valor": None})
    elif "ultimos 7 dias" in texto:
        filtros.append({"campo": "fechaCreacion", "operador": "ultimos_dias", "valor": 7})
    elif "ultimos 30 dias" in texto:
        filtros.append({"campo": "fechaCreacion", "operador": "ultimos_dias", "valor": 30})
    elif "ultimos 3 meses" in texto:
        filtros.append({"campo": "fechaCreacion", "operador": "ultimos_meses", "valor": 3})
    
    # Estados
    if "pendientes" in texto:
        filtros.append({"campo": "estadoInstancia", "operador": "=", "valor": "EN_CURSO"})
    elif "finalizados" in texto:
        filtros.append({"campo": "estadoInstancia", "operador": "=", "valor": "COMPLETADO"})
    elif "rechazados" in texto:
        filtros.append({"campo": "estadoInstancia", "operador": "=", "valor": "RECHAZADO"})
        
    return filtros

@router.post("/interpretar", response_model=ReporteResponse)
async def interpretar_reporte(req: ReporteRequest):
    load_resources()
    if model is None:
        raise HTTPException(status_code=500, detail="El modelo no está entrenado ni disponible.")

    # Preprocesamiento
    texto = req.texto.lower().strip()
    seq = tokenizer.texts_to_sequences([texto])
    padded = pad_sequences(seq, maxlen=30, padding='post', truncating='post')

    # Predicción
    preds = model.predict(padded)
    pred_intent = np.argmax(preds[0], axis=-1)[0]
    pred_format = np.argmax(preds[1], axis=-1)[0]
    pred_aclaracion = preds[2][0][0]

    intent_label = le_intent.inverse_transform([pred_intent])[0]
    format_label = le_format.inverse_transform([pred_format])[0]
    requiere_aclaracion = bool(pred_aclaracion > 0.5)

    confianza_intent = float(np.max(preds[0][0]))

    if intent_label == "ambiguo" or requiere_aclaracion:
        return ReporteResponse(
            requiereAclaracion=True,
            preguntaAclaratoria="¿Podrías dar más detalles sobre lo que deseas analizar? (Ej. agrupado por cliente, política, etc.)",
            confianza=confianza_intent
        )

    # Extraer de las reglas
    mapping = INTENT_MAPPINGS.get(intent_label, {})
    if not mapping:
        return ReporteResponse(
            requiereAclaracion=True,
            preguntaAclaratoria="No entiendo la intención del reporte.",
            confianza=confianza_intent
        )

    filtros = extract_filters(texto)
    
    # Construir respuesta
    response = ReporteResponse(
        titulo="Reporte Dinámico Generado",
        descripcion=f"Interpretación de: '{req.texto}'",
        intencionDetectada=intent_label,
        entidadPrincipal=mapping.get("entidadPrincipal"),
        metricas=[Metrica(**m) for m in mapping.get("metricas", [])],
        agrupaciones=mapping.get("agrupaciones", []),
        ordenamiento=[Ordenamiento(**o) for o in mapping.get("ordenamiento", [])],
        filtros=[Filtro(**f) for f in filtros],
        formatoSalida=format_label,
        visualizacion=mapping.get("visualizacion", "tabla"),
        confianza=confianza_intent,
        requiereAclaracion=False
    )

    return response

@router.post("/transcribir")
async def transcribir_audio(req: TranscripcionRequest):
    # Endpoint stub para la transcripción solicitada en el requerimiento
    # "Si no es viable implementar transcripción real ahora, dejar la estructura preparada"
    return {"textoTranscrito": "ejemplo de texto transcrito"}
