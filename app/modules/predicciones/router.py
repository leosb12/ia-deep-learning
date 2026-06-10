from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from typing import Dict, Any
import os

from app.modules.predicciones.config import settings
from app.modules.predicciones.schemas import DatasetRequest, PredictionRequest, PredictionResponse, ExplainRequest

router = APIRouter()

@router.get("/health")
async def health_check():
    return {
        "status": "ok",
        "modulo": "predicciones",
        "deepseek_configurado": bool(settings.DEEPSEEK_API_KEY)
    }

@router.get("/politicas/backend")
async def politicas_backend():
    from app.modules.predicciones.services.backend_client import get_politicas_reales
    try:
        politicas = await get_politicas_reales()
        return {"politicas": politicas}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error obteniendo politicas: {str(e)}")

@router.post("/dataset/generar-deepseek")
async def generar_dataset(req: DatasetRequest):
    from app.modules.predicciones.services.synthetic_dataset_service import generate_synthetic_dataset
    if not settings.DEEPSEEK_API_KEY:
        raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY no configurada")
    try:
        result = await generate_synthetic_dataset(req)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dataset/generar-local")
async def generar_dataset_masivo_local(req: DatasetRequest):
    """
    Genera miles de datos de entrenamiento en 1 segundo analizando 
    la estructura de la política y simulando los caminos localmente sin cobrar tokens.
    """
    from app.modules.predicciones.services.backend_client import get_politicas_reales
    from app.modules.predicciones.services.local_simulator_service import generar_dataset_local
    from app.modules.predicciones.services.dataset_storage_service import save_synthetic_dataset
    try:
        politicas = await get_politicas_reales()
        todos = []
        for pol in politicas:
            escenarios = generar_dataset_local(pol, req.cantidadPorPolitica)
            todos.extend(escenarios)
        
        csv_path, json_path = save_synthetic_dataset(todos)
        
        return {
            "mensaje": "Dataset masivo generado exitosamente",
            "politicas_analizadas": len(politicas),
            "escenarios_creados": len(todos),
            "csv_path": csv_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/sintetico")
async def ver_dataset_sintetico(page: int = 1, size: int = 10):
    from app.modules.predicciones.services.dataset_storage_service import get_synthetic_dataset
    data = get_synthetic_dataset(page, size)
    if data is None:
        raise HTTPException(status_code=404, detail="Dataset sintético no encontrado. Ejecuta /dataset/generar-deepseek primero.")
    return {"page": page, "size": size, "data": data}

@router.get("/dataset/sintetico/export/csv")
async def export_sintetico_csv():
    path = os.path.join(settings.DATASET_DIR, "synthetic_workflow_dataset.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")
    return FileResponse(path, media_type="text/csv", filename="synthetic_workflow_dataset.csv")

@router.get("/dataset/sintetico/export/json")
async def export_sintetico_json():
    path = os.path.join(settings.DATASET_DIR, "synthetic_workflow_dataset.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")
    return FileResponse(path, media_type="application/json", filename="synthetic_workflow_dataset.json")

@router.post("/dataset/combinar")
async def endpoint_combine_datasets():
    from app.modules.predicciones.services.dataset_storage_service import combine_datasets
    try:
        path = await combine_datasets()
        return {"message": "Datasets combinados exitosamente", "path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dataset/final")
async def ver_dataset_final(page: int = 1, size: int = 10):
    import pandas as pd
    path = os.path.join(settings.DATASET_DIR, "training_dataset_final.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset final no encontrado.")
    df = pd.read_csv(path)
    start = (page - 1) * size
    end = start + size
    data = df.iloc[start:end].to_dict(orient="records")
    return {"page": page, "size": size, "total": len(df), "data": data}

@router.get("/dataset/final/export/csv")
async def export_final_csv():
    path = os.path.join(settings.DATASET_DIR, "training_dataset_final.csv")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Dataset no encontrado.")
    return FileResponse(path, media_type="text/csv", filename="training_dataset_final.csv")

@router.post("/train")
def train():
    from app.modules.predicciones.services.training_service import train_models
    try:
        info = train_models()
        return {"message": "Entrenamiento completado", "models": info}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict", response_model=Any)
async def endpoint_predict(req: PredictionRequest):
    print("----- PREDICT CALL RECEIVED -----", req.politicaId)
    with open("predict_call.log", "a") as f:
        f.write(f"Call received for {req.politicaId}\nJSON:\n{req.politicaEstructuraJson}\n")
    try:
        from app.modules.predicciones.services.prediction_service import predict
        return await predict(req)
    except Exception as e:
        with open("error_predict.log", "w") as f:
            import traceback
            f.write(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/explicar")
async def endpoint_explicar(req: ExplainRequest):
    try:
        from app.modules.predicciones.services.explanation_service import get_explanation
        return await get_explanation(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
