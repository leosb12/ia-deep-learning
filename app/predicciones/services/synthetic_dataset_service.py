import logging
from app.predicciones.schemas import DatasetRequest
from app.predicciones.services.backend_client import get_politicas_reales
from app.predicciones.services.deepseek_client import generar_escenarios_deepseek
from app.predicciones.services.dataset_storage_service import save_synthetic_dataset

logger = logging.getLogger(__name__)

async def generate_synthetic_dataset(req: DatasetRequest) -> dict:
    # 1. Get real policies
    politicas = await get_politicas_reales()
    
    todos_los_escenarios = []
    motivos_descarte_global = []
    debug_global = None
    
    for pol in politicas:
        logger.info(f"Generando {req.cantidadPorPolitica} escenarios para la politica {pol.get('id', 'N/A')}")
        
        # Batch generation to prevent token limit (Unterminated string)
        batch_size = 20
        escenarios_faltantes = req.cantidadPorPolitica
        
        while escenarios_faltantes > 0:
            cantidad_lote = min(batch_size, escenarios_faltantes)
            logger.info(f"Generando lote de {cantidad_lote} escenarios (faltan {escenarios_faltantes})...")
            
            result_ds = await generar_escenarios_deepseek(pol, cantidad_lote)
            escenarios_pol = result_ds["escenarios"]
            debug_info = result_ds["debug"]
            
            # Save last debug info to show in response
            debug_global = debug_info
            
            for e in escenarios_pol:
                if not isinstance(e, dict):
                    motivos_descarte_global.append("El escenario no es un diccionario JSON.")
                    continue
                if "rutaEjecutadaCodificada" not in e:
                    motivos_descarte_global.append(f"Falta 'rutaEjecutadaCodificada' en el escenario. ({str(e)[:50]}...)")
                    continue
                todos_los_escenarios.append(e)
                
            if debug_info["motivos_descarte"]:
                motivos_descarte_global.extend(debug_info["motivos_descarte"])
                
            # Even if there was a JSON error, we discount the batch to avoid infinite loops
            escenarios_faltantes -= cantidad_lote
            
    # Combine errors
    if debug_global:
        debug_global["motivos_descarte"] = motivos_descarte_global
            
    # 2. Validate valid scenarios (we already filtered them above into todos_los_escenarios)
    escenarios_validos = todos_los_escenarios
    escenarios_recibidos = len(escenarios_validos) + len(motivos_descarte_global)
    
    if not escenarios_validos:
        logger.error("DeepSeek no devolvió ningún escenario válido. No se guardará el dataset.")
        raise ValueError(f"Fallo en generación de dataset. Debug: {debug_global}")
    
    # 3. Save
    csv_path, json_path = save_synthetic_dataset(escenarios_validos)
    
    return {
        "politicas_leidas": len(politicas),
        "escenarios_solicitados": len(politicas) * req.cantidadPorPolitica,
        "escenarios_recibidos": escenarios_recibidos,
        "escenarios_validos": len(escenarios_validos),
        "escenarios_descartados": len(motivos_descarte_global),
        "csv_path": csv_path,
        "json_path": json_path,
        "debug": debug_global
    }
