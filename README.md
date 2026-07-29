---
license: apache-2.0
language: en
pipeline_tag: text-to-speech
tags:
  - text-to-speech
  - speech-synthesis
  - vits
  - cpu
  - edge-ai
  - small-model
  - 24khz
  - voice-distillation
base_model: owensong/Inflect-Micro-v2
---

# Scrappy — a 9M-param CPU voice, distilled in one day

**Scrappy** is the voice of [ScrappyLabs](https://scrappylabs.ai): a full text-to-speech voice
in **9.36M parameters / 37MB**, running **10–14× faster than real-time on a plain CPU** —
no GPU, no cloud, 0.16s load. It's a fine-tune of
[owensong/Inflect-Micro-v2](https://huggingface.co/owensong/Inflect-Micro-v2) (Apache-2.0),
trained the same day that model crossed our radar.

Listen: [`samples/scrappy_intro.wav`](samples/scrappy_intro.wav) — Scrappy says hello
(generated on a desktop CPU by this exact checkpoint).

**Get it:** [🤗 HuggingFace](https://huggingface.co/scrappylabsai/scrappy-voice-1)
(canonical weights) · [🐙 GitHub](https://github.com/scrappylabsai/scrappy-voice-1)
(clone-and-run mirror, issues & PRs) · Story: [scrappylabs.ai](https://scrappylabs.ai)

## Why this exists

Inflect-Micro-v2 ships as an inference-only release — one fixed voice, no trainer, and the
docs say voice replacement is "research use." We wanted to know how true that is, so we
treated it as a one-day exercise: reconstruct the training stack, distill a voice we like
into it, and publish what we learned. This repo is the result **plus the missing piece we
had to build — a working fine-tuning setup** (see `trainer/`).

The recipe, end to end:

1. **Teacher renders the corpus.** A commercial cloud TTS narrator voice generated ~4,400
   short clips (~5.5h @ 24kHz) from a text corpus we control — so every transcript is known
   by construction.
2. **An ASR gate cleans it.** Every clip is round-tripped through speech recognition and
   scored against its transcript. This caught real poison: clips where the teacher API
   quietly injected a *spoken watermark* instead of the requested text. Signal checks
   (clipping/silence/duration) run alongside. 98.6% survived.
3. **Warm-start fine-tune.** The released `model.pth` initializes the generator; the
   posterior encoder and discriminator start fresh (the release runtime ships the full
   training-side model code — only losses, data loading, and the loop needed writing).
   Decoder frozen for the first 3k steps, LR 1e-4 → 5e-6 over 50k steps, batch 24, fp32.
4. **50,000 steps ≈ 3 hours on one rented RTX PRO 6000 Blackwell (~$9).** A 24GB laptop
   GPU does the same run in ~6 hours if you'd rather boil a laptop (we don't recommend it).

## Usage

Identical to upstream — this is a drop-in checkpoint for the packaged runtime:

```python
from inference import InflectTTS

tts = InflectTTS(model_dir=".", device="cpu")
tts.save("Hi, I'm Scrappy. I run on your CPU.", "out.wav", seed=7)
```

```bash
python inference.py --model-dir . --device cpu --text "Hello from Scrappy." --output out.wav
```

Notes carried over from upstream: English only, single voice, deterministic seeds,
punctuation-aware long-form chunking, `speed` 0.5–2.0, `variation` 0.0–1.0. Write numbers
out as words for best results.

## Fine-tune your own voice (`trainer/`)

The `trainer/` directory contains the training stack upstream deliberately omits:

- `prep_filelists.py` — phonemizes transcripts with the model's own frontend and validates
  every symbol against the release inventory (espeak emits out-of-vocabulary diacritics on
  exotic names — unvalidated, they crash training).
- `train_ft.py` — the full loop: VITS losses, torchaudio mel transforms (slaney/slaney —
  no librosa dependency), warm-start loading, decoder freeze schedule, drop-in candidate
  export every N steps.
- `eval_candidate.py` — renders fixed prompts from any candidate for A/B listening.

You also need the cython monotonic-alignment kernel from the
[canonical VITS repo](https://github.com/jaywalnut310/vits) (the release stubs it out) —
build it and drop the package into `runtime/`. Data contract: mono 24kHz clips, one speaker,
verified transcripts, 1–5+ hours. **Gate your corpus with ASR round-trips** — it's the only
check that catches audio that says the wrong thing beautifully.

## Honest limitations

- **Prosody is where distillation loses the most.** Timbre and identity transfer well;
  the teacher's long-range timing instincts (dramatic pauses, phrase-level planning) get
  averaged. The duration predictor is the smallest organ in a VITS — expect a flatter read
  than the source voice.
- Slight texture softness vs. a large vocoder remains at close listening.
- Everything upstream says about biases and English-only applies.

## Provenance & takedown

The training audio was synthesized by a commercial cloud TTS narrator voice (a synthetic
persona — no real person's voice was cloned). If you're a rights holder with a concern,
open a discussion on this repo and we'll respond promptly.

## Credits

- **[owensong/Inflect-Micro-v2](https://huggingface.co/owensong/Inflect-Micro-v2)** —
  base model, runtime, and an unusually honest set of docs (Apache-2.0)
- [VITS](https://github.com/jaywalnut310/vits) (MIT) — architecture lineage + alignment kernel
- Built in a day by [ScrappyLabs](https://scrappylabs.ai) — we do this kind of thing daily
  to stay sharp. Bring your own AI; we keep receipts.
