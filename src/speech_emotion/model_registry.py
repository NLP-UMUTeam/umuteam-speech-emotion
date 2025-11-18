import json
from pathlib import Path

DEFAULT_MODEL_CONFIG = {
    "es": { "text": "", "audio": "", "concat": "", "mean": "", "multihead": "" },
    "en": { "text": "", "audio": "", "concat": "", "mean": "", "multihead": "" }
}

def load_model_config(path=None):
    cfg = DEFAULT_MODEL_CONFIG.copy()

    if path and Path(path).is_file():
        with open(path, "r") as f:
            data = json.load(f)
        for lang in data:
            cfg[lang].update(data[lang])

    return cfg


def get_default_model_id(language, mode, config_path=None):
    config = load_model_config(config_path)
    return config[language][mode]
