from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import librosa
import soundfile as sf
import pyworld as pw
from transformers import pipeline as hf_pipeline
from jiwer import wer as jiwer_wer

from src.wavlm_scorer import WavLMEmbedder


class MetricCaches:
    def __init__(self, root: Path) -> None:
        self.emb_cache = root / "embeddings"; self.emb_cache.mkdir(parents=True, exist_ok=True)
        self.asr_cache = root / "asr"; self.asr_cache.mkdir(parents=True, exist_ok=True)
        self.feat_cache = root / "features"; self.feat_cache.mkdir(parents=True, exist_ok=True)


def file_hash(path: Path) -> str:
    st = path.stat()
    return hashlib.md5(f"{path}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()


def mfcc_20_mean(path: Path, target_sr: int = 16000, cache: MetricCaches | None = None) -> np.ndarray:
    out = None
    if cache:
        cf = cache.feat_cache / f"{file_hash(path)}.mfcc20.npy"
        if cf.exists():
            return np.load(cf)
        out = cf
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=int(0.025*sr), hop_length=int(0.010*sr))
    vec = mfcc.mean(axis=1)
    if out is not None:
        np.save(out, vec)
    return vec


def median_f0(path: Path, target_sr: int = 16000, cache: MetricCaches | None = None) -> float:
    if cache:
        cf = cache.feat_cache / f"{file_hash(path)}.f0.npy"
        if cf.exists():
            return float(np.load(cf))
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    _f0, t = pw.harvest(y.astype(np.float64), fs=sr)
    f0 = pw.stonemask(y.astype(np.float64), _f0, t, sr)
    med = float(np.median(f0[f0 > 0])) if np.any(f0 > 0) else 0.0
    if cache:
        np.save(cache.feat_cache / f"{file_hash(path)}.f0.npy", np.array(med))
    return med


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    d = (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / d) if d else 0.0


def wavlm_cosine(a: Path, b: Path, wavlm: WavLMEmbedder) -> float:
    ea = wavlm.embed_file(str(a)); eb = wavlm.embed_file(str(b))
    return cosine(ea, eb)


def mfcc_cosine(a: Path, b: Path, cache: MetricCaches | None) -> float:
    return cosine(mfcc_20_mean(a, cache=cache), mfcc_20_mean(b, cache=cache))


def pitch_similarity(a: Path, b: Path, cache: MetricCaches | None) -> float:
    f0a = median_f0(a, cache=cache); f0b = median_f0(b, cache=cache)
    if f0a <= 0 or f0b <= 0:
        return 0.0
    return max(0.0, 1.0 - abs(f0a - f0b) / max(f0a, f0b))


class ASREngine:
    def __init__(self, model_name: str = "openai/whisper-small") -> None:
        self.pipe = hf_pipeline("automatic-speech-recognition", model=model_name, device=-1)

    def transcribe(self, path: Path) -> str:
        return self.pipe(str(path))["text"]


def wer_score(ref_paths: list[Path], hyp_paths: list[Path], asr: ASREngine, cache: MetricCaches | None) -> float:
    n = min(len(ref_paths), len(hyp_paths))
    ref_texts, hyp_texts = [], []
    for i in range(n):
        r, h = ref_paths[i], hyp_paths[i]
        ref_texts.append(asr.transcribe(r))
        hyp_texts.append(asr.transcribe(h))
    if not ref_texts:
        return 0.0
    return float(jiwer_wer(ref_texts, hyp_texts))


