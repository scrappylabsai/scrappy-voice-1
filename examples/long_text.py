from pathlib import Path
import sys

MODEL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_DIR))

from inference import InflectTTS

text = (
    "Long input is divided at punctuation-aware boundaries. "
    "Each segment is generated locally, then joined with a controlled pause; "
    "this keeps memory bounded without requiring a remote service."
)
InflectTTS(MODEL_DIR, device="cpu").save(text, MODEL_DIR / "long_example.wav", seed=11)
