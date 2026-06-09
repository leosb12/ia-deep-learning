"""
Cliente del proveedor IA para el Motor de Reportes Inteligentes.
Internamente usa la configuración DeepSeek ya existente.
Externamente se presenta como "Motor IA Avanzado".

Este cliente SOLO envía prompts y recibe texto/JSON.
NUNCA accede a bases de datos ni ejecuta consultas.
"""
import httpx
import json
import logging
from typing import List, Dict, Any, Optional

from app.modules.reportes_dinamicos.config import reportes_settings

logger = logging.getLogger(__name__)


class MotorIAClient:
    """
    Cliente genérico del Motor IA Avanzado para interpretación semántica.
    Internamente usa la API configurada (DeepSeek u otro proveedor compatible OpenAI).
    """
    
    def __init__(self):
        self.api_key = reportes_settings.ia_api_key
        self.base_url = reportes_settings.ia_base_url
        self.model = reportes_settings.ia_model
        self.timeout = reportes_settings.ia_timeout_seconds
        self.temperature = reportes_settings.ia_temperature
        self.max_tokens = reportes_settings.ia_max_tokens
    
    def is_available(self) -> bool:
        """Verifica si el Motor IA Avanzado está configurado y disponible."""
        return bool(
            reportes_settings.ia_enabled 
            and self.api_key 
            and len(self.api_key) > 5
        )
    
    async def interpretar(self, messages: List[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """
        Envía mensajes al proveedor IA y retorna la respuesta parseada como JSON.
        
        Args:
            messages: Lista de mensajes [{role, content}]
            
        Returns:
            Dict parseado del JSON de respuesta, o None si falla.
        """
        if not self.is_available():
            logger.warning("Motor IA Avanzado no está disponible. Falta API key o está deshabilitado.")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        
        if self.max_tokens and self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        
        api_key_preview = self.api_key[:4] + "***" if self.api_key else "None"
        logger.info(f"Motor IA Avanzado: enviando solicitud ({api_key_preview})")
        
        try:
            timeout_val = max(self.timeout, 30.0)  # Mínimo 30 segundos
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=timeout_val
                )
                response.raise_for_status()
                
                response_data = response.json()
                content = response_data["choices"][0]["message"]["content"]
                
                logger.info(f"Motor IA Avanzado: respuesta recibida ({len(content)} chars)")
                
                # Limpiar posibles bloques markdown
                parsed = self._parse_json_response(content)
                return parsed
                
        except httpx.HTTPStatusError as e:
            logger.error(f"Motor IA Avanzado - Error HTTP: {e.response.status_code} - {e.response.text[:300]}")
            return None
        except httpx.TimeoutException:
            logger.error(f"Motor IA Avanzado - Timeout después de {timeout_val}s")
            return None
        except Exception as e:
            logger.error(f"Motor IA Avanzado - Error inesperado: {str(e)}")
            return None
    
    def _parse_json_response(self, content: str) -> Optional[Dict[str, Any]]:
        """Parsea la respuesta del proveedor IA, limpiando markdown si es necesario."""
        content = content.strip()
        
        # Limpiar bloques markdown
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        
        if content.endswith("```"):
            content = content[:-3]
        
        content = content.strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(f"Motor IA: Error parseando JSON: {e}")
            logger.debug(f"Motor IA: Contenido recibido: {content[:500]}")
            
            # Intento de recuperación: buscar JSON dentro del texto
            import re
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            return None


# Instancia singleton del cliente
motor_ia_client = MotorIAClient()
