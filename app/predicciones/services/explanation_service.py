from app.modules.predicciones.schemas import ExplainRequest
from app.modules.predicciones.services.deepseek_client import explicar_prediccion

async def get_explanation(req: ExplainRequest) -> dict:
    explanation_text = await explicar_prediccion(req.prediccion.model_dump())
    return {
        "explicacion": explanation_text,
        "nota": "Esta es una recomendación generada por IA y no modifica automáticamente el workflow."
    }
