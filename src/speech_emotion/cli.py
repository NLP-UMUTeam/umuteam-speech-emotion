import argparse
from .inference import predict_emotion

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--mode", default="text",
                   choices=["text","audio","concat","mean","multihead"])
    p.add_argument("--language", default="es", choices=["es","en"])
    p.add_argument("--model-id")
    p.add_argument("--model-config")
    p.add_argument("--device")

    args = p.parse_args()

    emotion = predict_emotion(
        audio_path=args.audio,
        model_id=args.model_id,
        mode=args.mode,
        language=args.language,
        device=args.device,
        model_config_path=args.model_config
    )

    print(f"Emotion detected is: {emotion}")
    

if __name__ == '__main__':
    main()
