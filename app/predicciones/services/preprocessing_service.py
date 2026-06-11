import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any
import pickle
import os
from app.modules.predicciones.config import settings

def preprocess_training_data(df: pd.DataFrame):
    # Categorical and numerical columns
    categorical_cols = ['politicaId', 'prioridadActual']
    numerical_cols = [
        'cantidadObservaciones', 'cantidadNodos', 'cantidadDecisiones', 
        'cantidadForks', 'cantidadJoins', 'cantidadRetornos', 'cantidadReprocesos', 
        'cantidadDocumentos', 'cantidadFuncionariosInvolucrados', 'duracionPromedioHistorica'
    ]
    
    # Fill missing values
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna("UNKNOWN").astype(str)
            
    for col in numerical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0).astype(float)
            
    # Scale numerical
    scaler = StandardScaler()
    scaled_num = scaler.fit_transform(df[numerical_cols])
    
    # Encode categorical
    encoders = {}
    encoded_cat = pd.DataFrame()
    for col in categorical_cols:
        le = LabelEncoder()
        # Add "UNKNOWN" to classes implicitly by including it in fit if we want, but simpler is:
        df[col] = df[col].astype(str)
        le.fit(df[col].tolist() + ["UNKNOWN"])
        encoded_cat[col] = le.transform(df[col])
        encoders[col] = le
        
    X = pd.concat([pd.DataFrame(scaled_num, columns=numerical_cols), encoded_cat], axis=1)
    
    label_encoders = {}
    y_dict = {}
    
    labels = ['riesgoDemoraLabel', 'cuelloBotellaLabel', 'anomaliaLabel', 'prioridadRecomendadaLabel', 'rutaRecomendadaLabel']
    for label in labels:
        if label in df.columns:
            df[label] = df[label].fillna("UNKNOWN" if df[label].dtype == object else 0)
            le = LabelEncoder()
            y_dict[label] = le.fit_transform(df[label].astype(str))
            label_encoders[label] = le
            
    os.makedirs(settings.MODEL_DIR, exist_ok=True)
    with open(os.path.join(settings.MODEL_DIR, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(settings.MODEL_DIR, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    with open(os.path.join(settings.MODEL_DIR, 'label_encoders.pkl'), 'wb') as f:
        pickle.dump(label_encoders, f)
        
    return X, y_dict, label_encoders

def preprocess_prediction_input(data: dict) -> pd.DataFrame:
    try:
        with open(os.path.join(settings.MODEL_DIR, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)
        with open(os.path.join(settings.MODEL_DIR, 'encoders.pkl'), 'rb') as f:
            encoders = pickle.load(f)
    except FileNotFoundError:
        raise ValueError("Model artifacts not found. Please train first.")
        
    df = pd.DataFrame([data])
    
    numerical_cols = [
        'cantidadObservaciones', 'cantidadNodos', 'cantidadDecisiones', 
        'cantidadForks', 'cantidadJoins', 'cantidadRetornos', 'cantidadReprocesos', 
        'cantidadDocumentos', 'cantidadFuncionariosInvolucrados', 'duracionPromedioHistorica'
    ]
    for col in numerical_cols:
        if col not in df.columns:
            df[col] = 0
            
    scaled_num = scaler.transform(df[numerical_cols])
    
    categorical_cols = ['politicaId', 'prioridadActual']
    encoded_cat = pd.DataFrame()
    for col in categorical_cols:
        le = encoders[col]
        val = str(df[col].iloc[0]) if col in df.columns else "UNKNOWN"
        if val not in le.classes_:
            val = "UNKNOWN" if "UNKNOWN" in le.classes_ else le.classes_[0]
        encoded_cat[col] = le.transform([val])
        
    X = pd.concat([pd.DataFrame(scaled_num, columns=numerical_cols), encoded_cat], axis=1)
    return X
