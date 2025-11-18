# Speech Emotion

`speech-emotion` is a lightweight Python package for emotion recognition from audio and text.  
It supports Spanish (`es`) and English (`en`) and provides several inference modes:

- **text** – Whisper transcription + text classifier  
- **audio** – Wav2Vec2-BERT audio classifier  
- **concat** – Multimodal fusion (audio + text) by concatenation  
- **mean** – Multimodal fusion by mean  
- **multihead** – Multimodal fusion using multi-head cross-attention  

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/NLP-UMUTeam/umuteam-speech-emotion
pip install -e .
```

This installs: 

- The python package `speech_emotion`
- The command-line tool `speech_emotion`

## Supported Emotions

### Spanish (6 labels)

```
anger, disgust, fear, joy, neutral, sadness
```

### English (7 labels)

```
angry, disgust, fear, happy, neutral, sad, surprise
```

## Command Line Usage

 ### Show help

```
speech-emotion -h
```

### Basic example (Spanish, text mode)

```
speech-emotion \
  --audio path/to/audio.wav \
  --language es \
  --mode text \
  --model-config model.json
```

### Audio-only (Wav2Vec2-BERT, English)

```
speech-emotion \
  --audio audio.wav \
  --language en \
  --mode audio \
  --model-config model.json
```

### Multimodal concat fusion

```
speech-emotion \
  --audio audio.wav \
  --language es \
  --mode concat \
  --model-config model.json
```

### Multimodal mean fusion

```
speech-emotion \
  --audio audio.wav \
  --language en \
  --mode mean \
  --model-config model.json
```

### Multimodal multi-head cross-attention fusion

```
speech-emotion \
  --audio audio.wav \
  --language es \
  --mode multihead \
  --model-config model.json
```

## Model Configuration 

Models are organized in a simple JSON file (`model.json`).
Each language defines the HuggingFace model IDs used for each inference mode.

The system uses **pretrained models developed by the UMUTeam**, which are published and available on **HuggingFace**.

```json
{
  "es": {
    "text": "UMUTeam/MarIA-emotion-es",
    "audio": "UMUTeam/w2v-bert-emotion-es",
    "concat": "UMUTeam/w2v-bert-beto-concat-emotion-es",
    "mean": "UMUTeam/w2v-bert-beto-mean-emotion-es",
    "multihead": "UMUTeam/w2v-bert-beto-multihead-emotion-es"
  },
  "en": {
    "text": "UMUTeam/roberta-emotion-en",
    "audio": "UMUTeam/w2v-bert-emotion-en",
    "concat": "UMUTeam/w2v-bert-beto-concat-emotion-en",
    "mean": "UMUTeam/w2v-bert-beto-mean-emotion-en",
    "multihead": "UMUTeam/w2v-bert-beto-multihead-emotion-en"
  }
}

```

## GPU Usage 

The system can run on CPU or GPU. You can control the device in two ways:

1.  **Automatically using CUDA**: If a GPU is available, the library will automatically use.

2. **Selecting the GPU with CUDA_VISIBLE_DEVICES**: To force the model to use only one GPU.

   ```
   CUDA_VISIBLE_DEVICES=0 speech-emotion \
     --audio audio.wav \
     --language en \
     --mode audio \
     --model-config model.json
   ```

3. **Passing the device manually through the CLI**

   ```
   speech-emotion \
     --audio audio.wav \
     --language es \
     --mode text \
     --device cuda:0 \
     --model-config model.json
   ```

## Repository Structure

```
speech_emotion/
│
├── model.json
├── README.md
├── pyproject.toml
├── src/
│   └── speech_emotion/
│       ├── cli.py
│       ├── inference.py
│       ├── config.py
│       ├── model_registry.py
│       ├── __init__.py
│       └── models/
│           ├── wav2vec2_bert_es.py
│           ├── wav2vec2_bert_en.py
│           ├── multimodal_es.py
│           ├── multimodal_en.py
│           ├── multimodal_multi_head_cross_attn_es.py
│           └── multimodal_multi_head_cross_attn_en.py

```

