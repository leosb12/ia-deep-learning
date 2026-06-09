import numpy as np
import pickle
from pathlib import Path
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

class PredictorReporteVisual:
    def __init__(self):
        self.model_path = MODELS_DIR / "reporte_visual_model.keras"
        self.tokenizer_path = MODELS_DIR / "tokenizer.pkl"
        self.le_tipo_path = MODELS_DIR / "label_encoder_tipo_grafico.pkl"
        self.le_intent_path = MODELS_DIR / "label_encoder_intencion.pkl"
        self.le_entidad_path = MODELS_DIR / "label_encoder_entidad.pkl"
        self.le_metrica_path = MODELS_DIR / "label_encoder_metrica.pkl"

        self.model = None
        self.tokenizer = None
        self.le_tipo = None
        self.le_intent = None
        self.le_entidad = None
        self.le_metrica = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return True
        try:
            if not self.model_path.exists():
                return False
            self.model = load_model(str(self.model_path))
            with open(self.tokenizer_path, 'rb') as f:
                self.tokenizer = pickle.load(f)
            with open(self.le_tipo_path, 'rb') as f:
                self.le_tipo = pickle.load(f)
            with open(self.le_intent_path, 'rb') as f:
                self.le_intent = pickle.load(f)
            with open(self.le_entidad_path, 'rb') as f:
                self.le_entidad = pickle.load(f)
            with open(self.le_metrica_path, 'rb') as f:
                self.le_metrica = pickle.load(f)
            self.loaded = True
            return True
        except Exception as e:
            print(f"Error cargando modelo de reportes visuales: {e}")
            return False

    def predecir_bloque(self, prompt: str) -> dict:
        if not self.load():
            return {}
        try:
            text = prompt.lower().strip()
            seq = self.tokenizer.texts_to_sequences([text])
            padded = pad_sequences(seq, maxlen=25, padding='post', truncating='post')
            
            preds = self.model.predict(padded, verbose=0)
            
            pred_tipo_idx = np.argmax(preds[0], axis=-1)[0]
            pred_intent_idx = np.argmax(preds[1], axis=-1)[0]
            pred_entidad_idx = np.argmax(preds[2], axis=-1)[0]
            pred_metrica_idx = np.argmax(preds[3], axis=-1)[0]

            tipo = self.le_tipo.inverse_transform([pred_tipo_idx])[0]
            intencion = self.le_intent.inverse_transform([pred_intent_idx])[0]
            entidad = self.le_entidad.inverse_transform([pred_entidad_idx])[0]
            metrica = self.le_metrica.inverse_transform([pred_metrica_idx])[0]

            return {
                "tipo": str(tipo),
                "intencion": str(intencion),
                "entidadPrincipal": str(entidad),
                "metrica": str(metrica),
                "confianza": float(np.max(preds[1][0]))
            }
        except Exception as e:
            print(f"Error al predecir bloque visual: {e}")
            return {}

predictor_visual = PredictorReporteVisual()
