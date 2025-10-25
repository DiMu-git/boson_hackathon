from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import librosa
import soundfile as sf
from transformers import AutoFeatureExtractor, AutoModel


class WavLMEmbedder:
    """
    WavLM embedding with simple mean-pooled last hidden state and on-disk caching.
    """

    def __init__(
        self,
        cache_dir: Path,
        device: Optional[str] = None,
        target_sample_rate: int = 16000,
        model_name: str = "microsoft/wavlm-base-plus",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr = target_sample_rate
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _cache_path_for(self, audio_path: Path) -> Path:
        stat = audio_path.stat()
        key = f"wavlm|{str(audio_path)}|{stat.st_size}|{int(stat.st_mtime)}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.wavlm.npy"

    def embed_file(self, audio_path: str) -> np.ndarray:
        p = Path(audio_path)
        out = self._cache_path_for(p)
        if out.exists():
            return np.load(out)

        data, sr = sf.read(str(p))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != self.target_sr:
            data = librosa.resample(y=data.astype(np.float32), orig_sr=sr, target_sr=self.target_sr)

        with torch.no_grad():
            inputs = self.feature_extractor(
                np.array(data), sampling_rate=self.target_sr, return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs, output_hidden_states=True)
            last_hidden = outputs.last_hidden_state  # [1, T, C]
            emb = last_hidden.mean(dim=1).squeeze(0).cpu().numpy()

        np.save(out, emb)
        return emb

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


