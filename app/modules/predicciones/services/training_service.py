import os
import pandas as pd
import json
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow as tf
from app.modules.predicciones.config import settings
from app.modules.predicciones.services.preprocessing_service import preprocess_training_data
import logging

logger = logging.getLogger(__name__)

def train_models():
    dataset_path = os.path.join(settings.DATASET_DIR, "training_dataset_final.csv")
    if not os.path.exists(dataset_path):
        raise ValueError("Dataset final no encontrado. Ejecuta /dataset/combinar primero.")
        
    df = pd.read_csv(dataset_path)
    if len(df) < 10:
        raise ValueError("Dataset muy pequeño para entrenar. Genera más datos.")
        
    X, y_dict, label_encoders = preprocess_training_data(df)
    
    input_dim = X.shape[1]
    
    models_info = {}
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    
    folder_map = {
        "cuelloBotellaLabel": "predictor_cuellos_botella",
        "rutaRecomendadaLabel": "predictor_mejor_ruta",
        "prioridadRecomendadaLabel": "predictor_prioridad",
        "anomaliaLabel": "predictor_anomalias",
        "riesgoDemoraLabel": "predictor_riesgo_demora"
    }
    
    for label_name, y_data in y_dict.items():
        num_classes = len(label_encoders[label_name].classes_)
        
        # Don't train route if only 1 route available
        if label_name == 'rutaRecomendadaLabel' and num_classes <= 1:
            logger.info("Solo hay 1 ruta disponible para entrenamiento, saltando modelo de rutas.")
            models_info[label_name] = {"classes": 1, "model_path": None, "note": "Only 1 class available"}
            continue
            
        inputs = Input(shape=(input_dim,))
        x = Dense(64, activation='relu')(inputs)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        
        if num_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
            loss = 'binary_crossentropy'
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
            # Convert targets to binary 0/1 for sigmoid
            y_train = pd.Series(y_data).apply(lambda v: 1 if v == 1 else 0).values
        else:
            outputs = Dense(num_classes, activation='softmax')(x)
            loss = 'sparse_categorical_crossentropy'
            model = Model(inputs=inputs, outputs=outputs)
            model.compile(optimizer='adam', loss=loss, metrics=['accuracy'])
            y_train = y_data
            
        logger.info(f"Entrenando modelo para {label_name} con {num_classes} clases.")
        model.fit(X, y_train, epochs=20, batch_size=32, verbose=0)
        
        folder_name = folder_map.get(label_name, "")
        folder_path = os.path.join(settings.MODEL_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        model_path = os.path.join(folder_path, f"{label_name}_model.keras")
        model.save(model_path)
        
        models_info[label_name] = {
            "classes": int(num_classes),
            "model_path": model_path
        }
        
    metadata_path = os.path.join(settings.MODEL_DIR, "training_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(models_info, f, indent=2)
        
    return models_info
