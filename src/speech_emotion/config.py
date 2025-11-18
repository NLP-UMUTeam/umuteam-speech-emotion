import torch

# Español: 6 clases
ES_ID2LABEL = {
    0: "anger",
    1: "disgust",
    2: "fear",
    3: "joy",
    4: "neutral",
    5: "sadness",
}

ES_LABEL2ID = {v: k for k, v in ES_ID2LABEL.items()}

# Inglés: 7 clases
EN_ID2LABEL = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "neutral",
    5: "sad",
    6: "surprise",
}

EN_LABEL2ID = {v: k for k, v in EN_ID2LABEL.items()}

# Mapas por idioma
LANG_ID2LABEL = {
    "es": ES_ID2LABEL,
    "en": EN_ID2LABEL,
}
LANG_LABEL2ID = {
    "es": ES_LABEL2ID,
    "en": EN_LABEL2ID,
}

BERT_MODELS = {
    "es": "dccuchile/bert-base-spanish-wwm-uncased",
    "en": "google-bert/bert-base-uncased",
}

DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"


def get_default_bert(language: str) -> str:
    language = language.lower()
    return BERT_MODELS[language]


def get_label_maps(language: str):
    language = language.lower()
    try:
        return LANG_LABEL2ID[language], LANG_ID2LABEL[language]
    except KeyError:
        raise ValueError(f"Idioma no soportado para labels: {language}")


def get_device(device=None):
    if device:
        return torch.device(device)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
