#!/usr/bin/env python3
"""survivors.tsv → phonemized train/val filelists (wav_path|phoneme_text).

Runs on the TRAINING box next to the Inflect-Micro-v2 package (uses its exact
frontend so training tokens match inference tokens).
"""
import argparse
import random
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]
PKG = BASE / "Inflect-Micro-v2"
sys.path.insert(0, str(PKG / "runtime"))
sys.path.insert(0, str(PKG))

from inflect_vits_frontend import run_vits_frontend_batch  # noqa: E402
from text.symbols import symbols  # noqa: E402

KNOWN = set(symbols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--survivors", default=str(BASE / "corpus/survivors.tsv"))
    ap.add_argument("--wavs", default=str(BASE / "corpus/wavs24k"))
    ap.add_argument("--val-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out-dir", default=str(BASE / "corpus"))
    args = ap.parse_args()

    rows = [l.rstrip("\n").split("\t", 1) for l in open(args.survivors) if "\t" in l]
    texts = [t for _, t in rows]
    print(f"phonemizing {len(rows)} lines via Inflect frontend…")
    outs = run_vits_frontend_batch(texts, jobs=4)

    entries, oov = [], 0
    for (fid, _), out in zip(rows, outs):
        wav = Path(args.wavs) / f"{fid}.wav"
        if not (wav.exists() and out.phoneme_text):
            continue
        bad = set(out.phoneme_text) - KNOWN
        if bad:
            oov += 1
            continue
        entries.append(f"{wav}|{out.phoneme_text}")
    if oov:
        print(f"dropped {oov} lines with out-of-vocabulary phoneme symbols")

    rng = random.Random(args.seed)
    rng.shuffle(entries)
    n_val = max(4, int(len(entries) * args.val_frac))
    val, train = entries[:n_val], entries[n_val:]

    out_dir = Path(args.out_dir)
    (out_dir / "filelist_train.txt").write_text("\n".join(train) + "\n")
    (out_dir / "filelist_val.txt").write_text("\n".join(val) + "\n")
    print(f"{len(train)} train / {len(val)} val → {out_dir}/filelist_*.txt")


if __name__ == "__main__":
    main()
