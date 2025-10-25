from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln


RANDOM_SEED = 2025


def pick_top_speakers(extracted_root: Path, top_n: int = 5) -> List[str]:
    counts: Dict[str, int] = {}
    for flac in extracted_root.rglob("*.flac"):
        spk = flac.parent.parent.name
        counts[spk] = counts.get(spk, 0) + 1
    return [spk for spk, _ in sorted(counts.items(), key=lambda kv: (-kv[1], int(kv[0])) )][:top_n]


def normalize_audio(wav: np.ndarray, sr: int, target_sr: int = 16000, target_lufs: float = -23.0) -> Tuple[np.ndarray, int]:
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    # trim silence to <200ms at both ends (energy-based)
    wav, _ = librosa.effects.trim(wav, top_db=40)
    # loudness normalize (fallback to RMS if fails)
    try:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav)
        gain = target_lufs - loudness
        wav = pyln.normalize.loudness(wav, loudness, target_lufs)
    except Exception:
        # RMS normalize to -20 dBFS equivalent
        rms = np.sqrt(np.mean(wav ** 2) + 1e-12)
        if rms > 0:
            wav = wav * (10 ** (-20 / 20)) / rms
    return wav.astype(np.float32), sr


def save_wav(path: Path, wav: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr)


