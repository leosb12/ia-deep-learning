import os
import shutil
import json

base_dir = "app/modules"
to_delete = ["detector_anomalias", "predictor_cuellos_botella", "predictor_mejor_ruta", "predictor_prioridad"]

# 1. Delete empty modules
for d in to_delete:
    path = os.path.join(base_dir, d)
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted {path}")

# 2. Map labels to folders
folder_map = {
    "cuelloBotellaLabel": "predictor_cuellos_botella",
    "rutaRecomendadaLabel": "predictor_mejor_ruta",
    "prioridadRecomendadaLabel": "predictor_prioridad",
    "anomaliaLabel": "predictor_anomalias",
    "riesgoDemoraLabel": "predictor_riesgo_demora"
}

models_dir = "app/modules/predicciones/models"
metadata_path = os.path.join(models_dir, "training_metadata.json")

# 3. Create folders and move files
if os.path.exists(metadata_path):
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    for label, folder in folder_map.items():
        folder_path = os.path.join(models_dir, folder)
        os.makedirs(folder_path, exist_ok=True)
        
        # Look for the file
        old_file = os.path.join(models_dir, f"{label}_model.keras")
        new_file = os.path.join(folder_path, f"{label}_model.keras")
        
        if os.path.exists(old_file):
            shutil.move(old_file, new_file)
            print(f"Moved {old_file} to {new_file}")
            
        if label in metadata:
            metadata[label]["model_path"] = new_file.replace('\\', '/')

    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
        print("Updated training_metadata.json")

# 4. Patch training_service.py so future trainings use the new folders
import re
train_svc_path = "app/modules/predicciones/services/training_service.py"
with open(train_svc_path, 'r', encoding='utf-8') as f:
    content = f.read()

replacement = '''
        folder_map = {
            "cuelloBotellaLabel": "predictor_cuellos_botella",
            "rutaRecomendadaLabel": "predictor_mejor_ruta",
            "prioridadRecomendadaLabel": "predictor_prioridad",
            "anomaliaLabel": "predictor_anomalias",
            "riesgoDemoraLabel": "predictor_riesgo_demora"
        }
        folder_name = folder_map.get(label_name, "")
        folder_path = os.path.join(settings.MODEL_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        model_path = os.path.join(folder_path, f"{label_name}_model.keras")
'''
if 'folder_map =' not in content:
    content = content.replace('model_path = os.path.join(settings.MODEL_DIR, f"{label_name}_model.keras")', replacement.strip('\n'))
    with open(train_svc_path, 'w', encoding='utf-8') as f:
        f.write(content)
        print("Patched training_service.py")

