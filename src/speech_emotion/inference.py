from __future__ import annotations

from typing import Dict, Tuple, Optional

import torch
import librosa
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
    BertModel,
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

import warnings
warnings.filterwarnings("ignore")

import transformers
transformers.logging.set_verbosity_error()

# ==================================================================
#                            CACHES
# ==================================================================

_bert_cache: Dict[Tuple[str, str], Tuple[BertModel, AutoTokenizer]] = {}
_whisper_cache: Dict[Tuple[str, str], "pipeline"] = {}


# ==================================================================
#                      LAZY IMPORT HELPERS
# ==================================================================

def _lazy_import_whisper():
    """
    Importa los componentes de Whisper solo cuando se necesitan.
    """
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline as hf_pipeline
    return AutoModelForSpeechSeq2Seq, AutoProcessor, hf_pipeline


def _lazy_import_wav2vec_cls(language: str):
    """
    Importa la clase CustomAudioClassification de wav2vec2_bert_* de forma perezosa.
    """
    if language == "es":
        from .models.wav2vec2_bert_es import CustomAudioClassification as Wav2Vec2BertES
        return Wav2Vec2BertES
    elif language == "en":
        from .models.wav2vec2_bert_en import CustomAudioClassification as Wav2Vec2BertEN
        return Wav2Vec2BertEN
    else:
        raise ValueError(f"Idioma no soportado para wav2vec2_bert: {language}")


def _lazy_import_multimodal_cls(language: str, variant: str):
    """
    Importa las clases multimodales (Concat, Mean, MultiHeadAttn) de forma perezosa.
    """
    if language == "es":
        if variant == "concat":
            from .models.multimodal_es import CustomAudioClassificationConcat as MultimodalConcatES
            return MultimodalConcatES
        elif variant == "mean":
            from .models.multimodal_es import CustomAudioClassificationMean as MultimodalMeanES
            return MultimodalMeanES
        elif variant == "multihead":
            from .models.multimodal_multi_head_cross_attn_es import CustomAudioClassificationAttn as MultimodalAttnES
            return MultimodalAttnES
    elif language == "en":
        if variant == "concat":
            from .models.multimodal_en import CustomAudioClassificationConcat as MultimodalConcatEN
            return MultimodalConcatEN
        elif variant == "mean":
            from .models.multimodal_en import CustomAudioClassificationMean as MultimodalMeanEN
            return MultimodalMeanEN
        elif variant == "multihead":
            from .models.multimodal_multi_head_cross_attn_en import CustomAudioClassificationAttn as MultimodalAttnEN
            return MultimodalAttnEN

    raise ValueError(f"Idioma/variante no soportados para multimodal: language={language}, variant={variant}")


# ==================================================================
#                            BERT CACHE
# ==================================================================

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
        AutoModelForSpeechSeq2Seq, AutoProcessor, hf_pipeline = _lazy_import_whisper()

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            DEFAULT_WHISPER_MODEL,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        ).to(device)

        processor = AutoProcessor.from_pretrained(DEFAULT_WHISPER_MODEL)
        _whisper_cache[key] = hf_pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
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
    audio_path: Optional[str],
    text: Optional[str],
    device: torch.device,
    language: str,
) -> dict:
    """
    Clasificación solo texto.

    - Si `text` no es None -> se usa directamente (NO se llama a Whisper).
    - Si `text` es None y hay `audio_path` -> se transcribe con Whisper.

    Devuelve:
        {
            "top_label": str,
            "top_score": float,
            "scores": {label: prob, ...}
        }
    """
    if text is not None:
        transcription = text
    else:
        if audio_path is None:
            raise ValueError("For mode='text' you must provide either audio_path or text.")
        transcription = get_transcription(audio_path, device, language)

    clf = pipeline(
        "text-classification",
        model=model_id,
        device=device,
        return_all_scores=True,
    )

    outputs = clf(transcription)[0]  # lista de dicts [{'label':..., 'score':...}, ...]

    scores = {o["label"]: float(o["score"]) for o in outputs}
    top = max(outputs, key=lambda x: x["score"])

    return {
        "top_label": top["label"],
        "top_score": float(top["score"]),
        "scores": scores,
    }


# ---------- Solo audio (Wav2Vec2 + clasificación) ----------

def get_w2vbert_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
) -> dict:
    """
    Clasificación solo audio (wav2vec2_bert_*).

    Devuelve:
        {
            "top_label": str,
            "top_score": float,
            "scores": {label: prob, ...}
        }
    """
    label2id, id2label = get_label_maps(language)

    audio_array, _ = librosa.load(audio_path, sr=16000)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    ModelCls = _lazy_import_wav2vec_cls(language)
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
        probs = torch.softmax(logits, dim=-1)[0]

    scores = {
        id2label[i]: float(probs[i].cpu()) for i in range(len(probs))
    }
    pred_id = int(torch.argmax(probs).cpu())

    return {
        "top_label": id2label[pred_id],
        "top_score": float(probs[pred_id].cpu()),
        "scores": scores,
    }


# ---------- Multimodal (audio + texto) ----------

def _generic_multimodal_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
    variant: str,  # "concat" | "mean" | "multihead"
    text: Optional[str] = None,
) -> dict:
    """
    Multimodal (audio + texto).

    - Siempre requiere `audio_path`.
    - Si `text` no es None -> se usa como transcripción (NO se llama a Whisper).
    - Si `text` es None -> se usa Whisper para transcribir el audio.

    Devuelve:
        {
            "top_label": str,
            "top_score": float,
            "scores": {label: prob, ...}
        }
    """
    if audio_path is None:
        raise ValueError("Multimodal modes require an audio_path.")

    label2id, id2label = get_label_maps(language)

    audio_array, _ = librosa.load(audio_path, sr=16000)

    if text is not None:
        transcription = text
    else:
        transcription = get_transcription(audio_path, device, language)

    sent_emb = get_sentence_embedding(transcription, device, language)
    sent_emb = torch.tensor(sent_emb).unsqueeze(0).to(device)

    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)

    ModelCls = _lazy_import_multimodal_cls(language, variant)
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
        probs = torch.softmax(logits, dim=-1)[0]

    scores = {
        id2label[i]: float(probs[i].cpu()) for i in range(len(probs))
    }
    pred_id = int(torch.argmax(probs).cpu())

    return {
        "top_label": id2label[pred_id],
        "top_score": float(probs[pred_id].cpu()),
        "scores": scores,
    }


def get_w2vbert_bert_concat_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
    text: Optional[str] = None,
) -> dict:
    return _generic_multimodal_emotion(
        model_id=model_id,
        audio_path=audio_path,
        device=device,
        language=language,
        variant="concat",
        text=text,
    )


def get_w2vbert_bert_mean_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
    text: Optional[str] = None,
) -> dict:
    return _generic_multimodal_emotion(
        model_id=model_id,
        audio_path=audio_path,
        device=device,
        language=language,
        variant="mean",
        text=text,
    )


def get_w2vbert_bert_multihead_emotion(
    model_id: str,
    audio_path: str,
    device: torch.device,
    language: str,
    text: Optional[str] = None,
) -> dict:
    return _generic_multimodal_emotion(
        model_id=model_id,
        audio_path=audio_path,
        device=device,
        language=language,
        variant="multihead",
        text=text,
    )


# ==================================================================
#                    API PRINCIPAL: predict_emotion
# ==================================================================

def predict_emotion(
    audio_path: Optional[str] = None,
    text: Optional[str] = None,
    model_id: Optional[str] = None,
    mode: str = "text",          # "text" | "audio" | "concat" | "mean" | "multihead"
    language: str = "es",        # "es" | "en"
    device: str | torch.device | None = None,
    model_config_path: Optional[str] = None,
) -> dict:
    """
    API de alto nivel.

    - mode="text":
        - se puede usar con `audio_path` (Whisper) o con `text` directo.
        - si solo hay texto (sin audio), este es el único modo permitido.
    - modos multimodales ("concat", "mean", "multihead"):
        - requieren `audio_path`.
        - pueden usar transcripción pasada en `text` para evitar Whisper.
    - mode="audio":
        - solo audio (wav2vec2_bert_*), ignora `text`.

    Devuelve:
        {
            "top_label": str,
            "top_score": float,
            "scores": {label: prob, ...}
        }
    """
    device = get_device(device)
    mode = mode.lower()
    language = language.lower()

    # Validaciones básicas
    if mode == "text":
        if audio_path is None and text is None:
            raise ValueError("For mode='text' you must provide either audio_path or text.")
    else:
        # audio / concat / mean / multihead
        if audio_path is None:
            raise ValueError(f"Mode='{mode}' requires an audio_path.")

    if model_id is None:
        model_id = get_default_model_id(language, mode, model_config_path)

    if mode == "text":
        return get_text_emotion(model_id, audio_path, text, device, language)
    elif mode == "audio":
        return get_w2vbert_emotion(model_id, audio_path, device, language)
    elif mode == "concat":
        return get_w2vbert_bert_concat_emotion(model_id, audio_path, device, language, text=text)
    elif mode == "mean":
        return get_w2vbert_bert_mean_emotion(model_id, audio_path, device, language, text=text)
    elif mode == "multihead":
        return get_w2vbert_bert_multihead_emotion(model_id, audio_path, device, language, text=text)
    else:
        raise ValueError(f"Modo no reconocido: {mode}")
