#!/usr/bin/env python3
"""Synthesize eval prompts from a training candidate (or the stock release).

Mirrors InflectTTS internals but loads an arbitrary candidate checkpoint before
weight-norm removal. Writes WAVs for the ASR judge + blind listening.
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

BASE = Path(__file__).resolve().parents[1]
PKG = BASE / "Inflect-Micro-v2"
sys.path.insert(0, str(PKG / "runtime"))
sys.path.insert(0, str(PKG))

import commons  # noqa: E402
import utils  # noqa: E402
from inference import edge_fade, optimize_for_inference  # noqa: E402
from inflect_vits_frontend import run_vits_frontend  # noqa: E402
from models import SynthesizerTrn  # noqa: E402
from text import cleaned_text_to_sequence  # noqa: E402
from text.symbols import symbols  # noqa: E402

PROMPTS = [
    ("held1", "The committee will publish its findings on the fourteenth of October."),
    ("held2", "Beneath the ice, the ocean kept its own kind of time."),
    ("held3", "First, check the seal. Second, note the pressure. Finally, log both numbers."),
    ("held4", "Was it courage, or simply the absence of any other option?"),
    ("held5", "The fleet in this house runs the same quality inference that once required a data center."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, help="candidate .pth (or the stock model.pth)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hps = utils.get_hparams_from_file(str(PKG / "config.json"))
    net = SynthesizerTrn(len(symbols), hps.data.filter_length // 2 + 1,
                         hps.train.segment_size // hps.data.hop_length,
                         **hps.model).to(device).eval()
    ck = torch.load(args.candidate, map_location="cpu", weights_only=True)
    net.load_state_dict(ck["model"], strict=True)
    optimize_for_inference(net)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "prompts.tsv").write_text("\n".join(f"{k}\t{t}" for k, t in PROMPTS) + "\n")
    with torch.inference_mode():
        for key, text in PROMPTS:
            phonemes = run_vits_frontend(text).phoneme_text
            seq = cleaned_text_to_sequence(phonemes)
            if hps.data.add_blank:
                seq = commons.intersperse(seq, 0)
            tokens = torch.LongTensor(seq).unsqueeze(0).to(device)
            lengths = torch.LongTensor([tokens.size(1)]).to(device)
            torch.manual_seed(args.seed)
            wav = net.infer(tokens, lengths, noise_scale=0.667, noise_scale_w=0.8,
                            length_scale=1.0)[0][0, 0].float().cpu().numpy()
            sf.write(out / f"{key}.wav", edge_fade(wav, hps.data.sampling_rate), hps.data.sampling_rate)
            print(f"{key}: {len(wav)/hps.data.sampling_rate:.1f}s")
    print(f"eval wavs → {out}")


if __name__ == "__main__":
    main()
