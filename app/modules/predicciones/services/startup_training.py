import os
import asyncio
import logging
from app.modules.predicciones.config import settings

logger = logging.getLogger(__name__)

async def auto_train_if_missing():
    # Verificar si existen los artefactos principales del modelo
    required_artifacts = [
        os.path.join(settings.MODEL_DIR, "scaler.pkl"),
        os.path.join(settings.MODEL_DIR, "encoders.pkl"),
        os.path.join(settings.MODEL_DIR, "label_encoders.pkl"),
        os.path.join(settings.MODEL_DIR, "training_metadata.json")
    ]
    
    missing_artifacts = [art for art in required_artifacts if not os.path.exists(art)]
    
    if not missing_artifacts:
        logger.info("Auto-training: Todos los artefactos del modelo ya existen. No se requiere entrenamiento.")
        return
        
    logger.info(f"Auto-training: Faltan artefactos ({len(missing_artifacts)}). Iniciando proceso de autoentrenamiento...")
    
    # Verificar si el dataset final existe, si no, generarlo
    dataset_path = os.path.join(settings.DATASET_DIR, "training_dataset_final.csv")
    if not os.path.exists(dataset_path):
        logger.info("Auto-training: training_dataset_final.csv no encontrado. Intentando generar dataset sintético...")
        
        from app.modules.predicciones.services.backend_client import get_politicas_reales
        from app.modules.predicciones.services.local_simulator_service import generar_dataset_local
        from app.modules.predicciones.services.dataset_storage_service import save_synthetic_dataset, combine_datasets
        
        # Intentar conectar con el backend para obtener las políticas reales.
        # Si el backend aún no está listo (por orden de dependencias en compose), reintentar varias veces.
        politicas = []
        for attempt in range(6):
            try:
                politicas = await get_politicas_reales()
                # POL-SOPORTE es el fallback si falla, lo que significa que el backend falló o no está listo.
                # Reintentamos si obtenemos el fallback y no es el último intento.
                if len(politicas) > 0 and politicas[0].get("id") != "POL-SOPORTE":
                    logger.info(f"Auto-training: Conexión exitosa con el backend. Obtenidas {len(politicas)} políticas reales.")
                    break
            except Exception as e:
                logger.warning(f"Auto-training: Intento {attempt + 1}/6 falló al contactar al backend: {e}")
            
            if attempt < 5:
                logger.info("Auto-training: Esperando 5 segundos a que el backend se levante...")
                await asyncio.sleep(5)
                
        # Si no se pudo obtener políticas reales tras los reintentos, se usará el fallback de get_politicas_reales() (POL-SOPORTE)
        if not politicas:
            politicas = await get_politicas_reales()
            
        try:
            logger.info("Auto-training: Generando escenarios de simulación local...")
            todos = []
            for pol in politicas:
                # Generar 200 escenarios por cada política
                escenarios = generar_dataset_local(pol, 200)
                todos.extend(escenarios)
            
            save_synthetic_dataset(todos)
            logger.info("Auto-training: Dataset sintético generado localmente. Combinando datasets...")
            await combine_datasets()
        except Exception as e:
            logger.error(f"Auto-training: Error crítico al generar/combinar el dataset: {e}")
            return

    # Entrenar los modelos
    from app.modules.predicciones.services.training_service import train_models
    try:
        logger.info("Auto-training: Comenzando el entrenamiento de los modelos TensorFlow...")
        # train_models se ejecuta de forma síncrona. 
        # Corremos en un ejecutor en segundo plano para no bloquear el event loop principal de FastAPI.
        loop = asyncio.get_running_loop()
        models_info = await loop.run_in_executor(None, train_models)
        logger.info(f"Auto-training: Modelos entrenados con éxito. Metadatos guardados: {list(models_info.keys())}")
    except Exception as e:
        logger.error(f"Auto-training: Error entrenando los modelos: {e}")
