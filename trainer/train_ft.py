#!/usr/bin/env python3
"""Warm-start fine-tune of Inflect-Micro-v2 on a new single-speaker corpus.

Model code comes from the release runtime/ (which ships the full training path:
SynthesizerTrn.forward, PosteriorEncoder, MultiPeriodDiscriminator). This file
adds the pieces the release deliberately omits: mel/STFT transforms, the VITS
losses, a data loader, and the loop.

Warm-start per upstream FINETUNING.md: generator init from model.pth
(enc_q + discriminator fresh), decoder frozen for the first N steps, low LR,
frequent inference-only candidate exports.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

BASE = Path(__file__).resolve().parents[1]
PKG = BASE / "Inflect-Micro-v2"
sys.path.insert(0, str(PKG / "runtime"))
sys.path.insert(0, str(PKG))

import commons  # noqa: E402
import utils  # noqa: E402
from models import MultiPeriodDiscriminator, SynthesizerTrn  # noqa: E402
from text import cleaned_text_to_sequence  # noqa: E402
from text.symbols import symbols  # noqa: E402

# ---------------------------------------------------------------- transforms
_mel_basis = {}
_hann = {}


def _window(win_size, device, dtype):
    key = f"{win_size}_{device}_{dtype}"
    if key not in _hann:
        _hann[key] = torch.hann_window(win_size).to(device=device, dtype=dtype)
    return _hann[key]


def spectrogram_torch(y, n_fft, hop_size, win_size):
    y = F.pad(y.unsqueeze(1), (int((n_fft - hop_size) / 2), int((n_fft - hop_size) / 2)),
              mode="reflect").squeeze(1)
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size,
                      window=_window(win_size, y.device, y.dtype),
                      center=False, pad_mode="reflect", normalized=False,
                      onesided=True, return_complex=True)
    return torch.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-6)


def _mel_fb(n_fft, num_mels, sr, fmin, fmax, device, dtype):
    key = f"{n_fft}_{num_mels}_{fmax}_{device}_{dtype}"
    if key not in _mel_basis:
        from torchaudio.functional import melscale_fbanks
        # slaney/slaney == librosa_mel_fn defaults used by upstream VITS
        fb = melscale_fbanks(n_fft // 2 + 1, fmin, fmax, num_mels, sr,
                             norm="slaney", mel_scale="slaney").T
        _mel_basis[key] = fb.to(device=device, dtype=dtype)
    return _mel_basis[key]


def spec_to_mel_torch(spec, n_fft, num_mels, sr, fmin, fmax):
    mel = torch.matmul(_mel_fb(n_fft, num_mels, sr, fmin, fmax, spec.device, spec.dtype), spec)
    return torch.log(torch.clamp(mel, min=1e-5))


def mel_spectrogram_torch(y, n_fft, num_mels, sr, hop_size, win_size, fmin, fmax):
    return spec_to_mel_torch(spectrogram_torch(y, n_fft, hop_size, win_size),
                             n_fft, num_mels, sr, fmin, fmax)


# ---------------------------------------------------------------- losses (VITS)
def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl.float().detach() - gl.float()))
    return loss * 2


def discriminator_loss(disc_real, disc_gen):
    loss = 0
    for dr, dg in zip(disc_real, disc_gen):
        loss += torch.mean((1 - dr.float()) ** 2) + torch.mean(dg.float() ** 2)
    return loss


def generator_loss(disc_outputs):
    loss = 0
    for dg in disc_outputs:
        loss += torch.mean((1 - dg.float()) ** 2)
    return loss


def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    z_p, logs_q = z_p.float(), logs_q.float()
    m_p, logs_p = m_p.float(), logs_p.float()
    z_mask = z_mask.float()
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    return torch.sum(kl * z_mask) / torch.sum(z_mask)


# ---------------------------------------------------------------- data
class TextAudioDataset(Dataset):
    def __init__(self, filelist, hps):
        self.items = [l.strip().split("|", 1) for l in open(filelist) if "|" in l]
        self.hps = hps

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        wav_path, phonemes = self.items[idx]
        seq = cleaned_text_to_sequence(phonemes)
        if self.hps.data.add_blank:
            seq = commons.intersperse(seq, 0)
        text = torch.LongTensor(seq)
        audio, sr = sf.read(wav_path, dtype="float32")
        assert sr == self.hps.data.sampling_rate, f"{wav_path}: {sr}"
        audio = torch.from_numpy(audio).float().unsqueeze(0)
        spec_path = wav_path.replace(".wav", ".spec.pt")
        try:
            spec = torch.load(spec_path, weights_only=True)
        except (FileNotFoundError, RuntimeError):
            spec = spectrogram_torch(audio, self.hps.data.filter_length,
                                     self.hps.data.hop_length, self.hps.data.win_length).squeeze(0)
            torch.save(spec, spec_path)
        return text, spec, audio.squeeze(0)


def collate(batch):
    batch = sorted(batch, key=lambda x: x[1].size(1), reverse=True)
    max_t = max(x[0].size(0) for x in batch)
    max_s = max(x[1].size(1) for x in batch)
    max_w = max(x[2].size(0) for x in batch)
    n = len(batch)
    text = torch.zeros(n, max_t, dtype=torch.long)
    spec = torch.zeros(n, batch[0][1].size(0), max_s)
    wav = torch.zeros(n, 1, max_w)
    tl, sl, wl = (torch.zeros(n, dtype=torch.long) for _ in range(3))
    for i, (t, s, w) in enumerate(batch):
        text[i, :t.size(0)] = t
        spec[i, :, :s.size(1)] = s
        wav[i, 0, :w.size(0)] = w
        tl[i], sl[i], wl[i] = t.size(0), s.size(1), w.size(0)
    return text, tl, spec, sl, wav, wl


# ---------------------------------------------------------------- export
def export_candidate(net_g, step, lr, out_dir, orig_ck_format):
    sd = {k: v for k, v in net_g.state_dict().items() if not k.startswith("enc_q.")}
    out = {"format": orig_ck_format, "model": sd, "iteration": step,
           "learning_rate": lr, "deployable_parameters": sum(v.numel() for v in sd.values())}
    path = Path(out_dir) / f"candidate_{step:06d}.pth"
    torch.save(out, path)
    return path


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-filelist", default=str(BASE / "corpus/filelist_train.txt"))
    ap.add_argument("--val-filelist", default=str(BASE / "corpus/filelist_val.txt"))
    ap.add_argument("--run-dir", default=str(BASE / "runs/pilot"))
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr-g", type=float, default=1e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--lr-gamma", type=float, default=0.9999)
    ap.add_argument("--freeze-dec-steps", type=int, default=1500)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--export-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--c-mel", type=float, default=45.0)
    ap.add_argument("--c-kl", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda")
    run_dir = Path(args.run_dir)
    (run_dir / "candidates").mkdir(parents=True, exist_ok=True)
    hps = utils.get_hparams_from_file(str(PKG / "config.json"))
    orig_ck = torch.load(PKG / "model.pth", map_location="cpu", weights_only=True)
    ck_format = orig_ck.get("format", "inflect_v2_inference_config_v1")

    model_cfg = dict(hps.model)
    model_cfg["inference_only"] = False
    seg_frames = hps.train.segment_size // hps.data.hop_length
    net_g = SynthesizerTrn(len(symbols), hps.data.filter_length // 2 + 1,
                           seg_frames, **model_cfg).to(device)
    net_d = MultiPeriodDiscriminator(model_cfg.get("use_spectral_norm", False)).to(device)

    missing, unexpected = net_g.load_state_dict(orig_ck["model"], strict=False)
    fresh = sorted({k.split(".")[0] for k in missing})
    print(f"warm-start: {len(orig_ck['model'])} tensors loaded, "
          f"fresh modules: {fresh}, unexpected: {len(unexpected)}")
    assert not unexpected, unexpected
    assert all(k.startswith("enc_q.") for k in missing), "unexpected missing keys"

    opt_g = torch.optim.AdamW(net_g.parameters(), args.lr_g, betas=(0.8, 0.99), eps=1e-9)
    opt_d = torch.optim.AdamW(net_d.parameters(), args.lr_d, betas=(0.8, 0.99), eps=1e-9)
    sched_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma=args.lr_gamma)
    sched_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma=args.lr_gamma)

    ds = TextAudioDataset(args.train_filelist, hps)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True,
                    collate_fn=collate, num_workers=4, persistent_workers=True)
    print(f"dataset: {len(ds)} clips, {len(dl)} steps/epoch, batch {args.batch_size}")

    net_g.dec.requires_grad_(False)
    dec_frozen = True
    print(f"decoder FROZEN for first {args.freeze_dec_steps} steps")

    step, t0 = 0, time.perf_counter()
    log = (run_dir / "train_log.jsonl").open("a")
    net_g.train()
    net_d.train()
    while step < args.max_steps:
        for text, tl, spec, sl, wav, wl in dl:
            if step >= args.max_steps:
                break
            if dec_frozen and step >= args.freeze_dec_steps:
                net_g.dec.requires_grad_(True)
                dec_frozen = False
                print(f"step {step}: decoder UNFROZEN")
            text, tl = text.to(device), tl.to(device)
            spec, sl = spec.to(device), sl.to(device)
            wav = wav.to(device)

            y_hat, l_length, attn, ids_slice, x_mask, z_mask, \
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(text, tl, spec, sl)

            mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                    hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
            y_mel = commons.slice_segments(mel, ids_slice, seg_frames)
            y_hat_mel = mel_spectrogram_torch(y_hat.squeeze(1), hps.data.filter_length,
                                              hps.data.n_mel_channels, hps.data.sampling_rate,
                                              hps.data.hop_length, hps.data.win_length,
                                              hps.data.mel_fmin, hps.data.mel_fmax)
            y = commons.slice_segments(wav, ids_slice * hps.data.hop_length, hps.train.segment_size)

            # discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
            loss_disc = discriminator_loss(y_d_hat_r, y_d_hat_g)
            opt_d.zero_grad()
            loss_disc.backward()
            commons.clip_grad_value_(net_d.parameters(), None)
            opt_d.step()

            # generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
            loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * args.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * args.c_kl
            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen = generator_loss(y_d_hat_g)
            loss_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
            opt_g.zero_grad()
            loss_all.backward()
            commons.clip_grad_value_(net_g.parameters(), None)
            opt_g.step()
            sched_g.step()
            sched_d.step()
            step += 1

            if step % args.log_every == 0:
                rate = step / (time.perf_counter() - t0)
                rec = {"step": step, "g": round(loss_all.item(), 3),
                       "mel": round(loss_mel.item(), 3), "dur": round(loss_dur.item(), 4),
                       "kl": round(loss_kl.item(), 3), "d": round(loss_disc.item(), 3),
                       "lr": sched_g.get_last_lr()[0], "steps_per_s": round(rate, 2)}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()

            if step % args.export_every == 0 or step == args.max_steps:
                utils.save_checkpoint(net_g, opt_g, sched_g.get_last_lr()[0], step,
                                      str(run_dir / "G_latest.pth"))
                utils.save_checkpoint(net_d, opt_d, sched_d.get_last_lr()[0], step,
                                      str(run_dir / "D_latest.pth"))
                p = export_candidate(net_g, step, sched_g.get_last_lr()[0],
                                     run_dir / "candidates", ck_format)
                print(f"exported {p}", flush=True)

    print("training complete")


if __name__ == "__main__":
    main()
