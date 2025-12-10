from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Any
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import base64
import tempfile
import os

app = FastAPI()

# CORS: permite llamadas desde tu HTML (localhost / 127.0.0.1)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1",
        "http://localhost",
        "http://127.0.0.1:9999",
        "http://localhost:9999",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_CONFIG_PATH = "../model.json"


# Servir directorio "/interface" como estáticos (CSS, JS, imágenes)
app.mount("/static", StaticFiles(directory="static"), name="static") 


def fake_emotion_scores(text: str | None = None):
    """
    Genera unas probabilidades falsas pero 'creíbles' para las emociones.
    Para las capturas de pantalla.
    """
    # Por simplicidad, dejamos joy como dominante.
    scores = {
        "anger":   0.05,
        "disgust": 0.03,
        "fear":    0.07,
        "joy":     0.60,
        "sadness": 0.10,
        "surprise":0.08,
        "neutral": 0.07,
    }
    
    # Dominante = la emoción con mayor score
    dominant = max(scores, key=scores.get)
    return dominant, scores


# Página principal
@app.get ("/", response_class = HTMLResponse)
def index():
    with open ("static/gui.html", "r", encoding="utf8") as f:
        return f.read ()


@app.post("/predict")
async def predict (request: Request):

    # Lazy loading
    from speech_emotion.inference import predict_emotion
    
    
    # Lead request
    content_type = request.headers.get ("content-type", "")
    body = await request.json ()
    
    print (body)
    
    emotion = predict_emotion (
        text = body['transcription'],
        language = body['language'],
        mode = body['mode'],
        model_config_path = MODEL_CONFIG_PATH
    )

    return emotion
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run ("app:app", host = "127.0.0.1", port = 8000, reload = True)
