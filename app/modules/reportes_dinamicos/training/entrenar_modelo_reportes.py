import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "datasets" / "dataset_reportes.csv"

print("Cargando dataset...")
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=['texto_usuario'])

# Variables
vocab_size = 5000
embedding_dim = 64
max_length = 30
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# X data
sentences = df['texto_usuario'].values

# Tokenizer
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Y data (Multiple Outputs)
le_intent = LabelEncoder()
y_intent = le_intent.fit_transform(df['intencion'].values)

le_format = LabelEncoder()
y_format = le_format.fit_transform(df['formatoSalida'].astype(str).values)

y_aclaracion = (df['requiereAclaracion'].astype(str) == 'True').astype(int).values

# Split data
X_train, X_test, y_intent_train, y_intent_test, y_format_train, y_format_test, y_aclaracion_train, y_aclaracion_test = train_test_split(
    padded, y_intent, y_format, y_aclaracion, test_size=0.2, random_state=42
)

# Model
input_layer = keras.layers.Input(shape=(max_length,))
embedding = keras.layers.Embedding(vocab_size, embedding_dim)(input_layer)
lstm = keras.layers.Bidirectional(keras.layers.LSTM(64))(embedding)
dense1 = keras.layers.Dense(64, activation='relu')(lstm)

out_intent = keras.layers.Dense(len(le_intent.classes_), activation='softmax', name='intent')(dense1)
out_format = keras.layers.Dense(len(le_format.classes_), activation='softmax', name='format')(dense1)
out_aclaracion = keras.layers.Dense(1, activation='sigmoid', name='aclaracion')(dense1)

model = keras.Model(inputs=input_layer, outputs=[out_intent, out_format, out_aclaracion])

model.compile(loss={'intent': 'sparse_categorical_crossentropy', 
                    'format': 'sparse_categorical_crossentropy',
                    'aclaracion': 'binary_crossentropy'},
              optimizer='adam',
              metrics={'intent': 'accuracy', 'format': 'accuracy', 'aclaracion': 'accuracy'})

print("Entrenando modelo...")
model.fit(X_train, 
          {'intent': y_intent_train, 'format': y_format_train, 'aclaracion': y_aclaracion_train},
          epochs=5,
          validation_data=(X_test, {'intent': y_intent_test, 'format': y_format_test, 'aclaracion': y_aclaracion_test}),
          batch_size=128)

print("Guardando artefactos...")
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
model.save(str(MODELS_DIR / "modelo_reportes.keras"))

with open(MODELS_DIR / 'tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(MODELS_DIR / 'label_encoders.pkl', 'wb') as handle:
    pickle.dump({
        'intent': le_intent,
        'format': le_format
    }, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Evaluar
loss, intent_loss, format_loss, ac_loss, intent_acc, format_acc, ac_acc = model.evaluate(X_test, {'intent': y_intent_test, 'format': y_format_test, 'aclaracion': y_aclaracion_test}, verbose=0)
metricas = {
    "intent_accuracy": float(intent_acc),
    "format_accuracy": float(format_acc),
    "aclaracion_accuracy": float(ac_acc)
}
with open(MODELS_DIR / 'metricas.json', 'w') as f:
    json.dump(metricas, f)

print(f"Entrenamiento completado y modelo guardado en {MODELS_DIR}")
