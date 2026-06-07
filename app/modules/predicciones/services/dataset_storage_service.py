import pandas as pd
import json
import os
import httpx
from io import StringIO
from app.modules.predicciones.config import settings
import logging

logger = logging.getLogger(__name__)

def save_synthetic_dataset(escenarios: list):
    os.makedirs(settings.DATASET_DIR, exist_ok=True)
    json_path = os.path.join(settings.DATASET_DIR, "synthetic_workflow_dataset.json")
    csv_path = os.path.join(settings.DATASET_DIR, "synthetic_workflow_dataset.csv")
    
    # Load existing if any, and append
    existing = []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                existing = json.load(f)
        except Exception:
            existing = []
    
    combined = existing + escenarios
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
        
    df = pd.DataFrame(combined)
    df.to_csv(csv_path, index=False)
    return csv_path, json_path

def get_synthetic_dataset(page: int = 1, size: int = 10):
    json_path = os.path.join(settings.DATASET_DIR, "synthetic_workflow_dataset.json")
    if not os.path.exists(json_path):
        return None
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    start = (page - 1) * size
    end = start + size
    return data[start:end]

async def combine_datasets():
    os.makedirs(settings.DATASET_DIR, exist_ok=True)
    synthetic_csv_path = os.path.join(settings.DATASET_DIR, "synthetic_workflow_dataset.csv")
    final_csv_path = os.path.join(settings.DATASET_DIR, "training_dataset_final.csv")
    
    # 1. Fetch real dataset from backend
    real_df = pd.DataFrame()
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(settings.BACKEND_DATASET_URL, timeout=30)
            response.raise_for_status()
            csv_data = StringIO(response.text)
            real_df = pd.read_csv(csv_data)
            
            # Normalize origin
            def map_origin(inst_id):
                if pd.isna(inst_id):
                    return "REAL"
                if str(inst_id).startswith("SIM-"):
                    return "SIMULADO_BACKEND"
                return "REAL"
            
            if 'origenDatos' not in real_df.columns and 'instanciaId' in real_df.columns:
                real_df['origenDatos'] = real_df['instanciaId'].apply(map_origin)
            
    except Exception as e:
        logger.error(f"Failed to fetch real dataset: {e}")

    # 2. Load synthetic dataset
    synth_df = pd.DataFrame()
    if os.path.exists(synthetic_csv_path):
        synth_df = pd.read_csv(synthetic_csv_path)
    
    if real_df.empty and synth_df.empty:
        raise ValueError("No data available to combine.")
    
    # 3. Combine
    final_df = pd.concat([real_df, synth_df], ignore_index=True)
    
    # 4. Clean empty and drop duplicates
    final_df.dropna(how='all', inplace=True)
    final_df.drop_duplicates(inplace=True)
    
    final_df.to_csv(final_csv_path, index=False)
    return final_csv_path
