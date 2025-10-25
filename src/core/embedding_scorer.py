from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import librosa
import soundfile as sf
import torchaudio  # Some environments lack backend API; stub if missing

# Stub missing torchaudio backend APIs to satisfy SpeechBrain checks
if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends_stub():
        return []
    torchaudio.list_audio_backends = _list_audio_backends_stub  # type: ignore[attr-defined]

if not hasattr(torchaudio, "get_audio_backend"):
    def _get_audio_backend_stub():
        return "soundfile"
    torchaudio.get_audio_backend = _get_audio_backend_stub  # type: ignore[attr-defined]

from speechbrain.pretrained import EncoderClassifier


class SpeakerEmbedder:
    """
    ECAPA-TDNN speaker embedding with simple on-disk caching.
    """

    def __init__(
        self,
        cache_dir: Path,
        device: Optional[str] = None,
        target_sample_rate: int = 16000,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.target_sr = target_sample_rate
        # Load pretrained ECAPA from SpeechBrain Hub
        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            run_opts={"device": self.device},
        )

    def _cache_path_for(self, audio_path: Path) -> Path:
        stat = audio_path.stat()
        key = f"{str(audio_path)}|{stat.st_size}|{int(stat.st_mtime)}"
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return self.cache_dir / f"{h}.npy"

    def embed_file(self, audio_path: str) -> np.ndarray:
        p = Path(audio_path)
        out = self._cache_path_for(p)
        if out.exists():
            return np.load(out)

        # Read with soundfile, resample with librosa to 16k mono
        data, sr = sf.read(str(p))
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != self.target_sr:
            data = librosa.resample(y=data.astype(np.float32), orig_sr=sr, target_sr=self.target_sr)

        wav = torch.tensor(data, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            emb = self.model.encode_batch(wav).squeeze(0).squeeze(0).cpu().numpy()

        np.save(out, emb)
        return emb

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)


