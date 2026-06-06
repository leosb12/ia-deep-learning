import json
import logging
import re
from typing import Any

import httpx

from app.shared.config import settings

logger = logging.getLogger(__name__)


class DeepSeekPolicyAnalysisClient:
    def __init__(self) -> None:
        self.api_key = settings.deepseek_api_key
        self.base_url = settings.deepseek_base_url
        self.model = settings.deepseek_model
        self.timeout_seconds = settings.deepseek_timeout_seconds
        self.max_policies = max(1, settings.deepseek_max_policies)

    @property
    def disponible(self) -> bool:
        return bool(settings.deepseek_enabled and self.api_key)

    def debe_analizar(self, texto: str, resultado_base: dict[str, Any], solicitado: bool = False) -> bool:
        if not self.disponible:
            return False
        if solicitado:
            return True
        if not settings.deepseek_auto_analysis:
            return False

        confianza = float(resultado_base.get("confianza", 0.0))
        requiere_mas_informacion = bool(resultado_base.get("requiereMasInformacion", False))
        contiene_datos = bool(re.search(r"\d|:|#|n.{0,2}mero|cliente|plan|requisito", texto, re.IGNORECASE))
        texto_extenso = len(texto.split()) >= 10
        return (
            requiere_mas_informacion
            or confianza < settings.deepseek_auto_confidence_threshold
            or contiene_datos
            or texto_extenso
        )

    def analizar(
        self,
        texto: str,
        politicas_candidatas: list[Any],
        resultado_base: dict[str, Any],
    ) -> dict[str, Any] | None:
        if not self.disponible:
            return None

        politicas = politicas_candidatas[: self.max_policies]
        politicas_por_id = {str(politica.id): politica for politica in politicas}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self._system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "textoUsuario": texto,
                            "resultadoBaseDeepLearning": resultado_base,
                            "politicasRealesCandidatas": [
                                self._serializar_politica(politica) for politica in politicas
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
            "max_tokens": 900,
            "stream": False,
        }

        try:
            response = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            contenido = response.json()["choices"][0]["message"]["content"]
            analisis = json.loads(contenido)
        except Exception:
            logger.exception("No se pudo obtener analisis estructurado desde DeepSeek")
            return None

        return self._normalizar_analisis(analisis, politicas_por_id)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "Eres un asistente de interpretacion para un backend de politicas. "
            "Debes responder exclusivamente en json valido. No ejecutes politicas, no inventes "
            "politicas y no devuelvas IDs fuera de politicasRealesCandidatas. Tu tarea es extraer "
            "intencion, datos mencionados, requisitos detectados y campos faltantes. "
            "IMPORTANTE: En 'informacionEntregada' y 'requisitosDetectados', usa los nombres exactos de los campos "
            "definidos en 'requisitosIniciales' de las políticas candidatas (por ejemplo, 'codigoCliente', 'cedula', 'nombre', etc.) "
            "como valor de 'campo' o 'nombre'. "
            "Si no hay una politica clara entre las candidatas, usa null en politicaSugeridaId. Formato json: "
            '{"intencionPrincipal":"string|null","politicaSugeridaId":"string|null",'
            '"politicaSugeridaNombre":"string|null","confianza":0.0,'
            '"informacionEntregada":[{"campo":"string","valor":"string","textoOriginal":"string"}],'
            '"requisitosDetectados":[{"nombre":"string","valor":"string","politicaId":"string|null"}],'
            '"camposFaltantes":[{"nombre":"string","motivo":"string"}],'
            '"observaciones":["string"]}'
        )

    @staticmethod
    def _serializar_politica(politica: Any) -> dict[str, Any]:
        return {
            "id": str(politica.id),
            "nombre": politica.nombre,
            "descripcion": politica.descripcion,
            "categoria": politica.categoria,
            "descripcionClasificacion": politica.descripcionClasificacion,
            "palabrasClave": politica.palabrasClave,
            "intencionesEjemplo": politica.intencionesEjemplo,
            "requisitosSugeridos": politica.requisitosSugeridos,
            "requisitosIniciales": [
                {
                    "nombre": req.nombre,
                    "label": req.label,
                    "tipo": req.tipo,
                    "obligatorio": req.obligatorio,
                }
                for req in getattr(politica, "requisitosIniciales", [])
            ],
        }

    def _normalizar_analisis(self, analisis: dict[str, Any], politicas_por_id: dict[str, Any]) -> dict[str, Any]:
        politica_id = self._texto_o_none(analisis.get("politicaSugeridaId"))
        if politica_id not in politicas_por_id:
            politica_id = None

        politica_nombre = None
        if politica_id:
            politica_nombre = politicas_por_id[politica_id].nombre

        return {
            "intencionPrincipal": self._texto_o_none(analisis.get("intencionPrincipal")),
            "politicaSugeridaId": politica_id,
            "politicaSugeridaNombre": politica_nombre,
            "confianza": self._confianza(analisis.get("confianza")),
            "informacionEntregada": self._normalizar_items(
                analisis.get("informacionEntregada"),
                ("campo", "valor", "textoOriginal"),
            ),
            "requisitosDetectados": self._normalizar_requisitos(
                analisis.get("requisitosDetectados"),
                politicas_por_id,
            ),
            "camposFaltantes": self._normalizar_items(
                analisis.get("camposFaltantes"),
                ("nombre", "motivo"),
            ),
            "observaciones": self._normalizar_lista_texto(analisis.get("observaciones")),
        }

    def _normalizar_requisitos(self, value: Any, politicas_por_id: dict[str, Any]) -> list[dict[str, str | None]]:
        requisitos = []
        for item in (value if isinstance(value, list) else []):
            if not isinstance(item, dict):
                continue
            politica_id = self._texto_o_none(item.get("politicaId"))
            requisitos.append(
                {
                    "nombre": self._texto(item.get("nombre")),
                    "valor": self._texto(item.get("valor")),
                    "politicaId": politica_id if politica_id in politicas_por_id else None,
                }
            )
        return [item for item in requisitos if item["nombre"] or item["valor"]]

    def _normalizar_items(self, value: Any, campos: tuple[str, ...]) -> list[dict[str, str]]:
        items = []
        for item in (value if isinstance(value, list) else []):
            if not isinstance(item, dict):
                continue
            normalizado = {campo: self._texto(item.get(campo)) for campo in campos}
            if any(normalizado.values()):
                items.append(normalizado)
        return items

    def _normalizar_lista_texto(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return [texto for texto in (self._texto(item) for item in value) if texto]

    @staticmethod
    def _texto(value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()[:300]

    def _texto_o_none(self, value: Any) -> str | None:
        texto = self._texto(value)
        if not texto or texto.lower() == "null":
            return None
        return texto

    @staticmethod
    def _confianza(value: Any) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 4)
        except (TypeError, ValueError):
            return 0.0
