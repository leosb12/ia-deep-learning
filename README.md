# ia-deep-learning-service

Microservicio dedicado a modelos deep learning propios del sistema de politicas de negocio.

Este servicio existe para aislar dependencias pesadas como TensorFlow en Python 3.10, sin obligar al `ia-service` principal a migrar de runtime ni a instalar librerias de deep learning.

## Arquitectura

Cada caso de uso deep learning vive como un modulo independiente dentro de `app/modules/`.

Modulos actuales:

- `clasificador_solicitudes`: CU-35, clasificacion de solicitudes con dos rutas:
  - modelo propio Keras entrenado con dataset local de clases conocidas;
  - clasificador dinamico semantico contra politicas reales recibidas en cada request.

Modulos reservados para futuras versiones:

- `predictor_prioridad`
- `predictor_cuellos_botella`
- `detector_anomalias`
- `predictor_mejor_ruta`

## Ejecutar localmente

```powershell
py -3.10 -m venv venv310
.\venv310\Scripts\Activate.ps1
pip install -r requirements.txt
python -m app.modules.clasificador_solicitudes.training.train_clasificador_politicas
uvicorn app.main:app --reload --port 8010
```

## Endpoints

```http
GET /health
POST /api/ia/modelo-propio/clasificar
POST /api/deep-learning/clasificador-solicitudes/clasificar
POST /api/deep-learning/clasificador-solicitudes/clasificar-dinamico
```

Body de ejemplo:

```json
{
  "texto": "mi internet esta muy lento"
}
```

## Relacion con otros servicios

`ia-service` sigue siendo el orquestador general de IA y puede llamar a este servicio por HTTP cuando necesite modelos deep learning.

Spring Boot tambien puede integrarse directamente contra este microservicio usando el endpoint del modulo correspondiente.

## CU-35: modelo propio vs clasificador dinamico

El modelo propio (`/api/ia/modelo-propio/clasificar`) usa TensorFlow/Keras con `Tokenizer`, `Embedding`, `Bidirectional LSTM`, `Dense`, `Dropout` y `Softmax`. Sirve para demostrar dataset propio, entrenamiento deep learning y prediccion sobre clases conocidas. Como cualquier clasificador Softmax cerrado, no puede predecir una politica nueva que no estuvo en entrenamiento.

El endpoint real para politicas dinamicas es `/api/deep-learning/clasificador-solicitudes/clasificar-dinamico`. Recibe el texto del usuario y la lista de politicas activas reales enviadas por Spring Boot desde MongoDB. Construye una representacion semantica por politica con nombre, descripcion, categoria, descripcion de clasificacion, palabras clave, intenciones ejemplo y requisitos sugeridos. Luego compara embeddings semanticos con `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` usando similitud coseno.

Opcionalmente, el endpoint dinamico puede pedir apoyo a DeepSeek para interpretar texto libre y devolver un `analisisDeepSeek` estructurado con intencion principal, informacion entregada, requisitos detectados y campos faltantes. DeepSeek no reemplaza el clasificador dinamico: solo analiza las politicas reales candidatas recibidas en el request y cualquier `politicaSugeridaId` se valida contra esos IDs antes de responder.

Para activarlo por request, configure `DEEPSEEK_ENABLED=true`, `DEEPSEEK_API_KEY=...` y envie `usarDeepSeek: true` en el body dinamico. Para activarlo automaticamente en mensajes largos, con datos o con baja confianza, use ademas `DEEPSEEK_AUTO_ANALYSIS=true`.

Esta clasificacion dinamica usa embeddings deep learning; no depende de `codigoClasificacion`, no hace mapping fijo contra MongoDB y no usa keywords como mecanismo principal. Por eso puede sugerir politicas futuras creadas por administradores sin reentrenar el modelo propio.

TensorFlow queda aislado en este servicio. `ia-service` solo orquesta por HTTP y Spring Boot traduce el resultado a datos reales del negocio, validando que el `politicaId` devuelto exista en MongoDB antes de responder al movil.

## Docker

```powershell
docker build -t ia-deep-learning-service .
docker run --rm -p 8010:8010 ia-deep-learning-service
```
