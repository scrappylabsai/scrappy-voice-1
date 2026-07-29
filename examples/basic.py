from pathlib import Path
import sys

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from inference import InflectTTS

tts = InflectTTS(MODEL_DIR, device="cpu")
tts.save(
    "A small local voice can still be useful.",
    MODEL_DIR / "example.wav",
    speed=1.0,
    variation=0.667,
    seed=7,
)
