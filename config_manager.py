# config_manager.py
import json
import os

SETTINGS_FILE = "settings.json"

# Configuración por defecto (si no existe el archivo)
DEFAULT_CONFIG = {
    "cub_200_url": "https://s3.amazonaws.com/fast-ai-imageclas/CUB_200_2011.tgz",
    "yolo_base_model": "yolov8m.pt",
    "images_per_species_default": 200,
    "epochs_default": 50
}


def load_config():
    """Carga la configuración desde el JSON o devuelve la default."""
    if not os.path.exists(SETTINGS_FILE):
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG

    try:
        with open(SETTINGS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return DEFAULT_CONFIG


def save_config(new_config):
    """Guarda el diccionario de configuración en el archivo JSON."""
    with open(SETTINGS_FILE, "w") as f:
        json.dump(new_config, f, indent=4)