import argparse
from .inference import predict_emotion

def main():
    p = argparse.ArgumentParser(description="Speech Emotion CLI")

    p.add_argument("--audio", help="Path to audio file (required for modes: audio, concat, mean, multihead).")
    p.add_argument("--text", help="Optional text transcript. If provided with mode=text, Whisper is skipped.")
    p.add_argument(
        "--mode",
        default="text",
        choices=["text", "audio", "concat", "mean", "multihead"],
        help="Inference mode.",
    )
    p.add_argument(
        "--language",
        default="es",
        choices=["es", "en"],
        help="Language of the models.",
    )
    p.add_argument("--model-id", help="Optional explicit model id (HF repo or local path).")
    p.add_argument("--model-config", help="Path to model.json with default models.")
    p.add_argument("--device", help='Device to use: "cpu", "cuda", "cuda:0", etc.')

    args = p.parse_args()

    # Validaciones de entrada
    if args.mode == "text":
        if args.audio is None and args.text is None:
            p.error("For mode='text' you must provide either --audio or --text.")
    else:
        # audio / concat / mean / multihead
        if args.audio is None:
            p.error(f"Mode '{args.mode}' requires --audio.")

    emotion = predict_emotion(
        audio_path=args.audio,
        text=args.text,
        model_id=args.model_id,
        mode=args.mode,
        language=args.language,
        device=args.device,
        model_config_path=args.model_config,
    )

    print(f"Emotion detected is: {emotion}")


if __name__ == "__main__":
    main()
