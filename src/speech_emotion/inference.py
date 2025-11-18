from __future__ import annotations

import torch
import librosa
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    BertModel,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)
from huggingface_hub import hf_hub_download

from .config import (
    DEFAULT_WHISPER_MODEL,
    get_default_bert,
    get_device,
    get_label_maps,
)
from .model_registry import get_default_model_id

# ====== IMPORTS SEGÚN IDIOMA (carpeta models/) ======
# models/wav2vec2_bert_es.py / wav2vec2_bert_en.py
from .models.wav2vec2_bert_es import CustomAudioClassification as Wav2Vec2BertES
from .models.wav2vec2_bert_en import CustomAudioClassification as Wav2Vec2BertEN

# models/multimodal_es.py / multimodal_en.py
from .models.multimodal_es import (
    CustomAudioClassificationConcat as MultimodalConcatES,
    CustomAudioClassificationMean as MultimodalMeanES,
)
from .models.multimodal_en import (
    CustomAudioClassificationConcat as MultimodalConcatEN,
    CustomAudioClassificationMean as MultimodalMeanEN,
)

# models/multimodal_multi_head_cross_attn_es.py / ..._en.py
from .models.multimodal_multi_head_cross_attn_es import (
    CustomAudioClassificationAttn as MultimodalAttnES,
)
from .models.multimodal_multi_head_cross_attn_en import (
    CustomAudioClassificationAttn as MultimodalAttnEN,
)

import warnings
warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()
# ==================================================================
#                            CACHES
# ==================================================================

_bert_cache: dict[tuple[str, str], tuple[BertModel, AutoTokenizer]] = {}
_whisper_cache: dict[tuple[str, str], pipeline] = {}


def _get_bert(language: str, device: torch.device):
    """
    Devuelve (modelo BERT, tokenizer) cacheados por (language, device).
    """
    key = (language, device.type)
    if key not in _bert_cache:
        model_name = get_default_bert(language)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(device)
        _bert_cache[key] = (model, tokenizer)
    return _bert_cache[key]


def _get_whisper(device: torch.device, language: str):
    """
    Devuelve el pipeline de Whisper cacheado por (device, language).
    """
    key = (device.type, language)
    if key not in _whisper_cache:
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            DEFAULT_WHISPER_MODEL,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        processor = AutoProcessor.from_pretrained(DEFAULT_WHISPER_MODEL)
        _whisper_cache[key] = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device=device,
            generate_kwargs={"task": "transcribe", "language": language},
        )
    return _whisper_cache[key]


def _load_state_from_hf(model_id: str, filename: str, device: torch.device):
    """
    Carga un fichero de pesos (por ejemplo 'model.bin') desde un repo de Hugging Face.
    """
    path = hf_hub_download(repo_id=model_id, filename=filename)
    return torch.load(path, map_location=device)


# ==================================================================
#                            BLOQUES BÁSICOS
# ==================================================================

def get_transcription(audio_path: str, device: torch.device, language: str) -> str:
    whisper = _get_whisper(device, language)
    result = whisper(audio_path)
    return result["text"].strip()


def get_sentence_embedding(text: str, device: torch.device, language: str):
    bert_model, tokenizer = _get_bert(language, device)
    encoded = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=512,
        return_tensors="pt",
    ).to(device)
    with torch.no_grad():
        emb = bert_model(**encoded).last_hidden_state[0, 0, :].cpu().tolist()
    return emb


# ==================================================================
#                            MODOS
# ==================================================================

# ---------- Solo texto (Whisper + modelo de texto HF) ----------

def get_text_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
) -> str:
    text = get_transcription(audio_path, device, language)
    clf = pipeline("text-classification", model=model_id, device=device)
    out = clf(text)
    return out[0]["label"]


# ---------- Solo audio (Wav2Vec2 + clasificación) ----------

def _get_wav2vec_cls(language: str):
    """
    Devuelve la clase CustomAudioClassification según idioma.
    """
    if language == "es":
        return Wav2Vec2BertES
    elif language == "en":
        return Wav2Vec2BertEN
    else:
        raise ValueError(f"Idioma no soportado para wav2vec2_bert: {language}")


def get_w2vbert_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
) -> str:
    label2id, id2label = get_label_maps(language)

    audio_array, sr = librosa.load(audio_path, sr=16000)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    ModelCls = _get_wav2vec_cls(language)
    model = ModelCls.from_pretrained(
        model_id,
        num_labels=len(label2id),
        num_extra_dims=768,
        label2id=label2id,
        id2label=id2label,
    ).to(device)

    try:
        state_dict = _load_state_from_hf(model_id, "model.bin", device)
        model.load_state_dict(state_dict)
    except Exception:
        pass

    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        max_length=16000,
        truncation=True,
        return_tensors="pt",
    )

    input_values = inputs["input_features"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
        pred_id = int(torch.argmax(logits, dim=1).cpu())
        pred_label = id2label[pred_id]

    return pred_label



# ---------- Multimodal (audio + texto) ----------

def _get_multimodal_classes(language: str):
    """
    Devuelve las clases (Concat, Mean, MultiHeadAttn) según idioma.
    """
    if language == "es":
        return MultimodalConcatES, MultimodalMeanES, MultimodalAttnES
    elif language == "en":
        return MultimodalConcatEN, MultimodalMeanEN, MultimodalAttnEN
    else:
        raise ValueError(f"Idioma no soportado para multimodal: {language}")


def _generic_multimodal_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
    variant: str,
) -> str:
    label2id, id2label = get_label_maps(language)

    audio_array, sr = librosa.load(audio_path, sr=16000)

    transcription = get_transcription(audio_path, device, language)
    sent_emb = get_sentence_embedding(transcription, device, language)
    sent_emb = torch.tensor(sent_emb).unsqueeze(0).to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    ConcatCls, MeanCls, AttnCls = _get_multimodal_classes(language)
    if variant == "concat":
        ModelCls = ConcatCls
    elif variant == "mean":
        ModelCls = MeanCls
    elif variant == "multihead":
        ModelCls = AttnCls
    else:
        raise ValueError(f"Variante multimodal no reconocida: {variant}")

    model = ModelCls.from_pretrained(
        model_id,
        num_labels=len(label2id),
        num_extra_dims=768,
        label2id=label2id,
        id2label=id2label,
    ).to(device)

    try:
        state_dict = _load_state_from_hf(model_id, "model.bin", device)
        model.load_state_dict(state_dict)
    except Exception:
        pass

    inputs = feature_extractor(
        audio_array,
        sampling_rate=16000,
        max_length=16000,
        truncation=True,
        return_tensors="pt",
    )

    input_values = inputs["input_features"].to(device)
    attention_mask = inputs.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(
            input_values,
            attention_mask=attention_mask,
            sentence_embedding=sent_emb,
        ).logits
        pred_id = int(torch.argmax(logits, dim=1).cpu())
        pred_label = id2label[pred_id]

    return pred_label



def get_w2vbert_bert_concat_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
) -> str:
    return _generic_multimodal_emotion(
        model_id=model_id,
        audio_path=audio_path,
        device=device,
        language=language,
        variant="concat",
    )


def get_w2vbert_bert_mean_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
) -> str:
    return _generic_multimodal_emotion(
        model_id=model_id,
        audio_path=audio_path,
        device=device,
        language=language,
        variant="mean",
    )


def get_w2vbert_bert_multihead_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
) -> str:
    return _generic_multimodal_emotion(
        model_id=model_id,
        audio_path=audio_path,
        device=device,
        language=language,
        variant="multihead",
    )


# ==================================================================
#                    API PRINCIPAL: predict_emotion
# ==================================================================

def predict_emotion(
    audio_path: str,
    model_id: str | None = None,
    mode: str = "text",          # "text" | "audio" | "concat" | "mean" | "multihead"
    language: str = "es",        # "es" | "en"
    device: str | torch.device | None = None,
    model_config_path: str | None = None,
) -> str:
    """
    API de alto nivel.

    - Si model_id es None, se coge del JSON (model_config_path) según (language, mode).
    - language elige:
        - modelo BERT (config.get_default_bert)
        - clases de models/*.py (es / en)
        - idioma de Whisper
    """
    device = get_device(device)
    mode = mode.lower()
    language = language.lower()

    if model_id is None:
        model_id = get_default_model_id(language, mode, model_config_path)

    if mode == "text":
        return get_text_emotion(model_id, audio_path, device, language)
    elif mode == "audio":
        return get_w2vbert_emotion(model_id, audio_path, device, language)
    elif mode == "concat":
        return get_w2vbert_bert_concat_emotion(model_id, audio_path, device, language)
    elif mode == "mean":
        return get_w2vbert_bert_mean_emotion(model_id, audio_path, device, language)
    elif mode == "multihead":
        return get_w2vbert_bert_multihead_emotion(model_id, audio_path, device, language)
    else:
        raise ValueError(f"Modo no reconocido: {mode}")
