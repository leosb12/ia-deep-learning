import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle
import json
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "datasets" / "reportes_visuales_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

print("Cargando dataset para reportes visuales...")
df = pd.read_csv(DATASET_PATH)
df = df.dropna(subset=['prompt'])

# Hyperparameters
vocab_size = 1000
embedding_dim = 32
max_length = 25
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"

# Input sentences
sentences = df['prompt'].values

# Tokenizer setup
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(sentences)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Label Encoders
le_tipo = LabelEncoder()
y_tipo = le_tipo.fit_transform(df['tipo_grafico'].values)

le_intent = LabelEncoder()
y_intent = le_intent.fit_transform(df['intencion'].values)

le_entidad = LabelEncoder()
y_entidad = le_entidad.fit_transform(df['entidad'].values)

le_metrica = LabelEncoder()
y_metrica = le_metrica.fit_transform(df['metrica'].values)

# Multi-output neural network model
input_layer = keras.layers.Input(shape=(max_length,))
embedding = keras.layers.Embedding(vocab_size, embedding_dim)(input_layer)
lstm = keras.layers.Bidirectional(keras.layers.LSTM(32))(embedding)
dense1 = keras.layers.Dense(32, activation='relu')(lstm)

out_tipo = keras.layers.Dense(len(le_tipo.classes_), activation='softmax', name='tipo_grafico')(dense1)
out_intent = keras.layers.Dense(len(le_intent.classes_), activation='softmax', name='intencion')(dense1)
out_entidad = keras.layers.Dense(len(le_entidad.classes_), activation='softmax', name='entidad')(dense1)
out_metrica = keras.layers.Dense(len(le_metrica.classes_), activation='softmax', name='metrica')(dense1)

model = keras.Model(inputs=input_layer, outputs=[out_tipo, out_intent, out_entidad, out_metrica])

model.compile(loss={'tipo_grafico': 'sparse_categorical_crossentropy', 
                    'intencion': 'sparse_categorical_crossentropy',
                    'entidad': 'sparse_categorical_crossentropy',
                    'metrica': 'sparse_categorical_crossentropy'},
              optimizer='adam',
              metrics={'tipo_grafico': 'accuracy', 'intencion': 'accuracy', 'entidad': 'accuracy', 'metrica': 'accuracy'})

print("Entrenando clasificador de reportes visuales...")
model.fit(padded, 
          {'tipo_grafico': y_tipo, 'intencion': y_intent, 'entidad': y_entidad, 'metrica': y_metrica},
          epochs=15,
          batch_size=16)

print("Guardando modelo y encoders serializados...")
model.save(str(MODELS_DIR / "reporte_visual_model.keras"))

with open(MODELS_DIR / 'tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(MODELS_DIR / 'label_encoder_tipo_grafico.pkl', 'wb') as handle:
    pickle.dump(le_tipo, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(MODELS_DIR / 'label_encoder_intencion.pkl', 'wb') as handle:
    pickle.dump(le_intent, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(MODELS_DIR / 'label_encoder_entidad.pkl', 'wb') as handle:
    pickle.dump(le_entidad, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(MODELS_DIR / 'label_encoder_metrica.pkl', 'wb') as handle:
    pickle.dump(le_metrica, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Proceso de entrenamiento e importación finalizado con éxito.")
