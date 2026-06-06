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



## Docker

```powershell
docker build -t ia-deep-learning-service .
docker run --rm -p 8010:8010 ia-deep-learning-service
```
