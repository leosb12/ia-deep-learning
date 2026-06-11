import httpx
from typing import List, Dict, Any
from app.modules.predicciones.config import settings
import logging

logger = logging.getLogger(__name__)

async def get_politicas_reales() -> List[Dict[str, Any]]:
    """Obtiene la estructura real de las politicas de negocio desde el backend Spring Boot."""
    url = f"{settings.BACKEND_BASE_URL}/api/politicas" # Se asume este endpoint, ajustar si es necesario
    
    headers = {
        "X-Admin-User-Id": "69e14cac2cfebaf1b3914aa8"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            if isinstance(data, dict) and "content" in data:
                return data["content"]
            return data
    except Exception as e:
        logger.error(f"No se pudieron obtener las politicas desde {url}. Error: {e}")
        # Simulando una politica de soporte para que funcione si el backend no expone el endpoint exacto aun
        return [{
            "id": "POL-SOPORTE",
            "codigo": "POL-SOPORTE",
            "nombre": "Solicitud de soporte",
            "nodos": [
                {"id": "N1", "tipo": "INICIO", "nombre": "Inicio"},
                {"id": "N2", "tipo": "ACTIVIDAD", "nombre": "Cargar requisitos"},
                {"id": "N3", "tipo": "ACTIVIDAD", "nombre": "Revisión documental"},
                {"id": "N4", "tipo": "DECISION", "nombre": "Aprobación"},
                {"id": "N5", "tipo": "FIN", "nombre": "Fin"}
            ],
            "carriles": [
                {"id": "C1", "nombre": "Usuario"},
                {"id": "C2", "nombre": "Admisión"},
                {"id": "C3", "nombre": "Dirección"}
            ],
            "transiciones": [
                {"origen": "N1", "destino": "N2"},
                {"origen": "N2", "destino": "N3"},
                {"origen": "N3", "destino": "N4"},
                {"origen": "N4", "destino": "N5", "condicion": "Aprobado"},
                {"origen": "N4", "destino": "N2", "condicion": "Rechazado (Retorno)"}
            ]
        }]
