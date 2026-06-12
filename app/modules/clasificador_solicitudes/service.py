import logging
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import HTTPException

from app.modules.clasificador_solicitudes.deepseek_client import DeepSeekPolicyAnalysisClient

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "clasificador_politicas.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METRICS_PATH = MODELS_DIR / "training_metrics.json"
ORIGEN_MODELO_PROPIO = "MODELO_PROPIO_DEEP_LEARNING"
ORIGEN_DINAMICO = "DEEP_LEARNING_SEMANTICO_DINAMICO"


class ModeloPropioService:
    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        tokenizer_path: Path = TOKENIZER_PATH,
        label_encoder_path: Path = LABEL_ENCODER_PATH,
        metrics_path: Path = METRICS_PATH,
    ) -> None:
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.label_encoder_path = label_encoder_path
        self.metrics_path = metrics_path
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._label_encoder: Any | None = None
        self._max_len: int | None = None

    def clasificar(self, texto: str) -> dict[str, Any]:
        texto_limpio = self._limpiar_texto(texto)
        if not texto_limpio:
            raise HTTPException(status_code=400, detail="El texto no puede estar vacio")

        self._cargar_componentes()

        from tensorflow.keras.preprocessing.sequence import pad_sequences

        secuencia = self._tokenizer.texts_to_sequences([texto_limpio])
        entrada = pad_sequences(secuencia, maxlen=self._max_len, padding="post", truncating="post")
        predicciones = self._model.predict(entrada, verbose=0)[0]

        top_indices = np.argsort(predicciones)[::-1][:3]
        top_resultados = [
            {
                "politicaId": str(self._label_encoder.inverse_transform([indice])[0]),
                "confianza": round(float(predicciones[indice]), 4),
            }
            for indice in top_indices
        ]

        return {
            "politicaId": top_resultados[0]["politicaId"],
            "confianza": top_resultados[0]["confianza"],
            "origen": ORIGEN_MODELO_PROPIO,
            "topResultados": top_resultados,
        }

    def _cargar_componentes(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._label_encoder is not None:
            return

        faltantes = [
            str(path)
            for path in (self.model_path, self.tokenizer_path, self.label_encoder_path, self.metrics_path)
            if not path.exists()
        ]
        if faltantes:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Modelo propio no disponible. Primero ejecuta el entrenamiento con "
                    "python -m app.modules.clasificador_solicitudes.training.train_clasificador_politicas. "
                    f"Archivos faltantes: {', '.join(faltantes)}"
                ),
            )

        try:
            import json

            from tensorflow.keras.models import load_model

            self._model = load_model(self.model_path)
            with self.tokenizer_path.open("rb") as archivo:
                self._tokenizer = pickle.load(archivo)
            with self.label_encoder_path.open("rb") as archivo:
                self._label_encoder = pickle.load(archivo)
            with self.metrics_path.open("r", encoding="utf-8") as archivo:
                metricas = json.load(archivo)
            self._max_len = int(metricas["max_len"])
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("No se pudo cargar el modelo propio de clasificacion")
            raise HTTPException(status_code=500, detail="Fallo la carga del modelo propio entrenado") from exc

    @staticmethod
    def _limpiar_texto(texto: str) -> str:
        texto = texto.lower().strip()
        texto = re.sub(r"[^a-záéíóúñü0-9\s]", " ", texto)
        texto = re.sub(r"\s+", " ", texto)
        return texto.strip()


def normalizar_para_busqueda(texto: str) -> str:
    texto = texto.lower()
    reemplazos = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ü': 'u', 'ñ': 'n'
    }
    for orig, rep in reemplazos.items():
        texto = texto.replace(orig, rep)
    texto = re.sub(r'[^a-z0-9\s]', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


def extraer_requisitos_reglas(texto: str, politicas: list[Any]) -> list[str]:
    texto_norm = normalizar_para_busqueda(texto)
    requisitos_detectados = set()

    todos_requisitos = {}
    for pol in politicas:
        reqs = getattr(pol, "requisitosIniciales", [])
        for req in reqs:
            todos_requisitos[req.nombre] = req

    sinonimos = {
        "codigoCliente": ["codigo de cliente", "codigo cliente", "cod cliente", "id cliente", "cliente", "identificador cliente", "nro cliente", "numero cliente"],
        "codigo_cliente": ["codigo de cliente", "codigo cliente", "cod cliente", "id cliente", "cliente", "identificador cliente", "nro cliente", "numero cliente"],
        "documentoIdentidad": ["documento de identidad", "documento identidad", "carnet", "carnet identidad", "ci", "cedula", "dni", "identidad", "cc"],
        "documento_identidad": ["documento de identidad", "documento identidad", "carnet", "carnet identidad", "ci", "cedula", "dni", "identidad", "cc"],
        "cedula": ["cedula", "documento", "dni", "cc", "identificacion", "ci"],
        "nombre": ["nombre", "usuario", "nombre completo", "me llamo"],
        "numeroFactura": ["factura", "numero de factura", "num factura", "nro factura", "factura nro"],
        "numero_factura": ["factura", "numero de factura", "num factura", "nro factura", "factura nro"],
        "direccion": ["direccion", "domicilio", "residencia"],
        "comprobanteDomicilio": ["comprobante de domicilio", "comprobante domicilio", "domicilio", "direccion", "factura luz", "factura agua", "recibo luz", "recibo agua", "servicio basico", "servicios basicos"],
        "comprobante_domicilio": ["comprobante de domicilio", "comprobante domicilio", "domicilio", "direccion", "factura luz", "factura agua", "recibo luz", "recibo agua", "servicio basico", "servicios basicos"],
        "comprobantePago": ["comprobante de pago", "comprobante pago", "recibo pago", "pago", "deposito", "transferencia"],
        "comprobante_pago": ["comprobante de pago", "comprobante pago", "recibo pago", "pago", "deposito", "transferencia"],
        "planActual": ["plan actual", "mi plan", "plan que tengo"],
        "planDeseado": ["plan deseado", "plan nuevo", "plan que quiero", "plan a cambiar"]
    }

    for nombre_req, req in todos_requisitos.items():
        terminos = set()
        terminos.add(normalizar_para_busqueda(nombre_req))

        split_camel = re.sub(r'([a-z])([A-Z])', r'\1 \2', nombre_req)
        terminos.add(normalizar_para_busqueda(split_camel))

        split_snake = nombre_req.replace('_', ' ')
        terminos.add(normalizar_para_busqueda(split_snake))

        if getattr(req, "label", None):
            terminos.add(normalizar_para_busqueda(req.label))

        if nombre_req in sinonimos:
            for sin in sinonimos[nombre_req]:
                terminos.add(normalizar_para_busqueda(sin))

        for termino in terminos:
            if not termino:
                continue
            if len(termino) <= 3:
                pattern = r'\b' + re.escape(termino) + r'\b'
                if re.search(pattern, texto_norm):
                    requisitos_detectados.add(nombre_req)
                    break
            else:
                if termino in texto_norm:
                    requisitos_detectados.add(nombre_req)
                    break

    return list(requisitos_detectados)


def calcular_ranking_politicas(
    politicas: list[Any],
    similitudes: np.ndarray,
    requisitos_detectados: list[str],
    usar_solo_requisitos_iniciales: bool = False,
) -> dict[str, Any]:
    max_similitud = float(max(similitudes)) if len(similitudes) > 0 else 0.0
    tiene_intencion_clara = max_similitud >= 0.45
    tiene_requisitos = len(requisitos_detectados) > 0

    if usar_solo_requisitos_iniciales:
        origen = "REQUISITOS"
    elif tiene_requisitos and tiene_intencion_clara:
        origen = "MIXTO"
    elif tiene_requisitos and not tiene_intencion_clara:
        origen = "REQUISITOS"
    else:
        origen = "INTENCION"

    resultados_calculados = []
    for i, politica in enumerate(politicas):
        score_semantico = round(float(max(0.0, min(1.0, similitudes[i]))), 4)

        reqs_politica = [r.nombre for r in getattr(politica, "requisitosIniciales", [])]
        coincidentes = [r for r in reqs_politica if r in requisitos_detectados]
        faltantes = [r for r in reqs_politica if r not in requisitos_detectados]

        if reqs_politica:
            score_requisitos = round(len(coincidentes) / len(reqs_politica), 4)
        else:
            score_requisitos = 0.0

        if origen == "MIXTO":
            score_final = round((score_semantico + score_requisitos) / 2.0, 4)
        elif origen == "REQUISITOS":
            score_final = score_requisitos
        else:
            score_final = score_semantico

        resultados_calculados.append({
            "politica": politica,
            "scoreSemantico": score_semantico,
            "scoreRequisitos": score_requisitos,
            "scoreFinal": score_final,
            "requisitosCoincidentes": coincidentes,
            "requisitosFaltantes": faltantes,
        })

    # Sort results
    resultados_calculados.sort(key=lambda x: (x["scoreFinal"], 0.0 if usar_solo_requisitos_iniciales else x["scoreSemantico"]), reverse=True)

    mejor_resultado = resultados_calculados[0]
    confianza_final = mejor_resultado["scoreFinal"]

    if origen == "INTENCION":
        requiere_mas_informacion = confianza_final < 0.45
    else:
        requiere_mas_informacion = len(mejor_resultado["requisitosFaltantes"]) > 0

    top_resultados = [
        {
            "politicaId": res["politica"].id,
            "nombrePolitica": res["politica"].nombre,
            "confianza": res["scoreFinal"],
            "scoreRequisitos": res["scoreRequisitos"],
            "scoreSemantico": res["scoreSemantico"],
            "scoreFinal": res["scoreFinal"],
            "requisitosCoincidentes": res["requisitosCoincidentes"],
            "requisitosFaltantes": res["requisitosFaltantes"],
        }
        for res in resultados_calculados[:3]
    ]

    return {
        "politicaId": mejor_resultado["politica"].id,
        "nombrePolitica": mejor_resultado["politica"].nombre,
        "confianza": confianza_final,
        "origen": origen,
        "metodoRecomendacion": origen,
        "requiereMasInformacion": requiere_mas_informacion,
        "requisitosDetectados": requisitos_detectados,
        "requisitosCoincidentes": mejor_resultado["requisitosCoincidentes"],
        "requisitosFaltantes": mejor_resultado["requisitosFaltantes"],
        "topResultados": top_resultados,
        "resultados_calculados": resultados_calculados,
    }


modelo_propio_service = ModeloPropioService()


class ClasificadorDinamicoService:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        deepseek_client: DeepSeekPolicyAnalysisClient | None = None,
    ) -> None:
        self.model_name = model_name
        self._model: Any | None = None
        self._deepseek_client = deepseek_client or DeepSeekPolicyAnalysisClient()

    def clasificar(
        self,
        texto: str,
        politicas: list[Any],
        usar_deepseek: bool = False,
        nombre_documento: str | None = None,
        usar_solo_requisitos_iniciales: bool = False,
    ) -> dict[str, Any]:
        texto_limpio = texto.strip()
        if not texto_limpio:
            raise HTTPException(status_code=400, detail="El texto no puede estar vacio")
        if not politicas:
            raise HTTPException(status_code=400, detail="Debe enviar al menos una politica activa")

        model = self._cargar_modelo()
        textos_politicas = [self._construir_texto_politica(politica) for politica in politicas]

        try:
            embeddings = model.encode(
                [texto_limpio, *textos_politicas],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )
        except Exception as exc:
            logger.exception("No se pudo calcular embeddings semanticos")
            raise HTTPException(status_code=503, detail="Fallo el modelo semantico dinamico") from exc

        solicitud_embedding = embeddings[0]
        politicas_embeddings = embeddings[1:]
        similitudes = np.matmul(politicas_embeddings, solicitud_embedding)

        # 1. Pass 1: Rule-based extraction and ranking
        requisitos_detectados = extraer_requisitos_reglas(texto_limpio, politicas)
        if nombre_documento:
            requisitos_doc = extraer_requisitos_reglas(nombre_documento, politicas)
            requisitos_detectados = list(set(requisitos_detectados).union(requisitos_doc))
            
        resultado = calcular_ranking_politicas(politicas, similitudes, requisitos_detectados, usar_solo_requisitos_iniciales)
        resultados_calculados = resultado.pop("resultados_calculados")

        # 2. Pass 2: DeepSeek analysis
        if self._deepseek_client.debe_analizar(texto_limpio, resultado, solicitado=usar_deepseek):
            politicas_candidatas = [res["politica"] for res in resultados_calculados]
            analisis_deepseek = self._deepseek_client.analizar(
                texto_limpio,
                politicas_candidatas,
                resultado,
            )
            if analisis_deepseek:
                resultado["analisisDeepSeek"] = analisis_deepseek
                requisitos_ds_names = set()
                for r in analisis_deepseek.get("requisitosDetectados", []):
                    if r.get("nombre"):
                        requisitos_ds_names.add(r["nombre"])
                for info in analisis_deepseek.get("informacionEntregada", []):
                    if info.get("campo"):
                        requisitos_ds_names.add(info["campo"])

                if requisitos_ds_names:
                    requisitos_detectados_actualizados = list(set(requisitos_detectados).union(requisitos_ds_names))
                    resultado_actualizado = calcular_ranking_politicas(politicas, similitudes, requisitos_detectados_actualizados, usar_solo_requisitos_iniciales)
                    resultado_actualizado["analisisDeepSeek"] = analisis_deepseek
                    resultado = resultado_actualizado
                    resultado.pop("resultados_calculados", None)

                if analisis_deepseek.get("camposFaltantes"):
                    resultado["requiereMasInformacion"] = True

        return resultado

    def _cargar_modelo(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
            return self._model
        except Exception as exc:
            logger.exception("No se pudo cargar el modelo sentence-transformers")
            raise HTTPException(status_code=503, detail="Modelo semantico dinamico no disponible") from exc

    def _construir_texto_politica(self, politica: Any) -> str:
        partes = [
            politica.nombre,
            politica.descripcion,
            f"Categoria: {politica.categoria}" if politica.categoria else None,
            politica.descripcionClasificacion,
            self._formatear_lista("Palabras clave", politica.palabrasClave),
            self._formatear_lista("Intenciones ejemplo", politica.intencionesEjemplo),
            self._formatear_lista("Requisitos", politica.requisitosSugeridos),
        ]
        return ". ".join(parte.strip() for parte in partes if parte and parte.strip())

    @staticmethod
    def _formatear_lista(etiqueta: str, valores: list[str]) -> str | None:
        if not valores:
            return None
        return f"{etiqueta}: {', '.join(valores)}"


clasificador_dinamico_service = ClasificadorDinamicoService()
