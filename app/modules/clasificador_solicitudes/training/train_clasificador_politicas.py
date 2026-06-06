import json
import pickle
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

BASE_DIR = Path(__file__).resolve().parents[1]
DATASET_PATH = BASE_DIR / "datasets" / "solicitudes_wifi_politicas.csv"
DATASET_AMPLIADO_PATH = BASE_DIR / "datasets" / "solicitudes_multidominio_politicas.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "clasificador_politicas.keras"
TOKENIZER_PATH = MODELS_DIR / "tokenizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"
METRICS_PATH = MODELS_DIR / "training_metrics.json"

COLUMNAS_REQUERIDAS = {"texto", "politicaId", "nombrePolitica"}
VOCAB_SIZE = 5000
MAX_LEN = 24
EMBEDDING_DIM = 64
RANDOM_STATE = 42


def limpiar_texto(texto: str) -> str:
    texto = str(texto).lower().strip()
    texto = re.sub(r"[^a-záéíóúñü0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto)
    return texto.strip()


def cargar_dataset() -> pd.DataFrame:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset local: {DATASET_PATH}")

    datasets = [pd.read_csv(DATASET_PATH)]
    if DATASET_AMPLIADO_PATH.exists():
        datasets.append(pd.read_csv(DATASET_AMPLIADO_PATH))
    dataset = pd.concat(datasets, ignore_index=True)
    columnas_faltantes = COLUMNAS_REQUERIDAS - set(dataset.columns)
    if columnas_faltantes:
        raise ValueError(f"El CSV debe incluir las columnas: {sorted(COLUMNAS_REQUERIDAS)}")

    dataset = dataset.dropna(subset=["texto", "politicaId", "nombrePolitica"]).copy()
    dataset["texto"] = dataset["texto"].map(limpiar_texto)
    dataset = dataset[dataset["texto"].str.len() > 0]

    conteo_clases = dataset["politicaId"].value_counts()
    clases_insuficientes = conteo_clases[conteo_clases < 2]
    if not clases_insuficientes.empty:
        raise ValueError(f"Cada politica necesita al menos 2 ejemplos: {clases_insuficientes.to_dict()}")

    return dataset


def construir_modelo(num_clases: int) -> Sequential:
    modelo = Sequential(
        [
            Input(shape=(MAX_LEN,)),
            Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
            Bidirectional(LSTM(64)),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(num_clases, activation="softmax"),
        ]
    )
    modelo.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return modelo


def main() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)

    dataset = cargar_dataset()
    textos = dataset["texto"].tolist()

    label_encoder = LabelEncoder()
    etiquetas = label_encoder.fit_transform(dataset["politicaId"])

    x_train_val_textos, x_test_textos, y_train_val, y_test = train_test_split(
        textos,
        etiquetas,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=etiquetas,
    )
    x_train_textos, x_val_textos, y_train, y_val = train_test_split(
        x_train_val_textos,
        y_train_val,
        test_size=0.15,
        random_state=RANDOM_STATE,
        stratify=y_train_val,
    )

    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer.fit_on_texts(x_train_textos)

    x_train = pad_sequences(
        tokenizer.texts_to_sequences(x_train_textos),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )
    x_test = pad_sequences(
        tokenizer.texts_to_sequences(x_test_textos),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )
    x_val = pad_sequences(
        tokenizer.texts_to_sequences(x_val_textos),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    modelo = construir_modelo(num_clases=len(label_encoder.classes_))
    early_stopping = EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True)
    historial = modelo.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=60,
        batch_size=16,
        callbacks=[early_stopping],
        verbose=1,
    )

    test_loss, test_accuracy = modelo.evaluate(x_test, y_test, verbose=0)

    metricas = {
        "train_accuracy_final": float(historial.history["accuracy"][-1]),
        "validation_accuracy_final": float(historial.history["val_accuracy"][-1]),
        "test_accuracy": float(test_accuracy),
        "test_loss": float(test_loss),
        "cantidad_ejemplos": int(len(dataset)),
        "cantidad_clases": int(len(label_encoder.classes_)),
        "fecha_entrenamiento": datetime.now(timezone.utc).isoformat(),
        "max_len": MAX_LEN,
        "tamano_vocabulario": min(VOCAB_SIZE, len(tokenizer.word_index) + 1),
        "vocab_size": VOCAB_SIZE,
        "clases_disponibles": label_encoder.classes_.tolist(),
        "nombres_clases": label_encoder.classes_.tolist(),
        "datasets": [str(DATASET_PATH.name), str(DATASET_AMPLIADO_PATH.name)] if DATASET_AMPLIADO_PATH.exists() else [str(DATASET_PATH.name)],
        "artefacto_final_entrenado_con_dataset_completo": True,
    }

    tokenizer_final = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
    tokenizer_final.fit_on_texts(textos)
    x_all = pad_sequences(
        tokenizer_final.texts_to_sequences(textos),
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    modelo_final = construir_modelo(num_clases=len(label_encoder.classes_))
    modelo_final.fit(
        x_all,
        etiquetas,
        epochs=max(35, len(historial.history["accuracy"])),
        batch_size=16,
        verbose=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    modelo_final.save(MODEL_PATH)
    with TOKENIZER_PATH.open("wb") as archivo:
        pickle.dump(tokenizer_final, archivo)
    with LABEL_ENCODER_PATH.open("wb") as archivo:
        pickle.dump(label_encoder, archivo)

    with METRICS_PATH.open("w", encoding="utf-8") as archivo:
        json.dump(metricas, archivo, ensure_ascii=False, indent=2)

    print("Entrenamiento finalizado")
    print(json.dumps(metricas, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
