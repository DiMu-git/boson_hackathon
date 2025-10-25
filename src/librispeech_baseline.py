from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
import itertools
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from datasets import load_dataset
import torchaudio
import tarfile
from collections import defaultdict
import soundfile as sf
from tqdm import tqdm

from src.voice_generator import VoiceGenerator
from src.voice_analyzer import VoiceAnalyzer
from src.embedding_scorer import SpeakerEmbedder
from src.wavlm_scorer import WavLMEmbedder
from jiwer import wer as jiwer_wer
from transformers import pipeline as hf_pipeline


@dataclass
class Split:
    train_paths: List[Path]
    test_paths: List[Path]
    path_to_transcript: Dict[Path, str]


def _choose_from_librispeech(dataset_root: Path, min_clips: int, seed: int, train_frac: float, speaker_id: Optional[str] = None) -> Tuple[str, Split]:
    cache_root = dataset_root / "data" / "librispeech_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    raw_root = dataset_root / "data" / "librispeech_raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    # Expect the archive to be present (downloaded earlier by torchaudio or manually)
    archives = list(raw_root.glob("*.tar.gz"))
    if not archives:
        # If not present, use torchaudio only to download the tar (no decoding)
        _ = torchaudio.datasets.LIBRISPEECH(
            root=str(raw_root),
            url="train-clean-100",
            download=True,
        )
        archives = list(raw_root.glob("*.tar.gz"))
        if not archives:
            raise RuntimeError("LibriSpeech archive not found after download.")

    archive = archives[0]
    extract_root = dataset_root / "data" / "librispeech_extracted"
    target_subset_dir = extract_root / "LibriSpeech" / "train-clean-100"

    if not target_subset_dir.exists():
        extract_root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(path=extract_root)

    # Build transcript mapping per chapter
    flac_files = list(target_subset_dir.rglob("*.flac"))
    if not flac_files:
        raise RuntimeError("No .flac files found after extracting LibriSpeech archive.")

    # Group by speaker (parent of chapter dir)
    speaker_to_files: Dict[str, List[Path]] = defaultdict(list)
    for f in flac_files:
        # Structure: .../LibriSpeech/train-clean-100/<speaker>/<chapter>/<utt>.flac
        try:
            speaker_id = f.parent.parent.name
        except Exception:
            continue
        speaker_to_files[speaker_id].append(f)

    # Choose requested speaker if provided, else pick a suitable one
    chosen = None
    if speaker_id and speaker_id in speaker_to_files:
        chosen = speaker_id
    else:
        candidates = {s: files for s, files in speaker_to_files.items() if len(files) >= min_clips}
        if candidates:
            chosen = max(candidates.keys(), key=lambda k: len(candidates[k]))
        else:
            if not speaker_to_files:
                raise RuntimeError("No speakers found in extracted LibriSpeech subset.")
            chosen = max(speaker_to_files.keys(), key=lambda k: len(speaker_to_files[k]))

    chosen_files = sorted(speaker_to_files[chosen])

    # Parse transcripts per chapter
    # Each chapter folder has a <speaker>-<chapter>.trans.txt
    transcript_cache: Dict[Path, Dict[str, str]] = {}

    def get_transcript_for(flac_path: Path) -> str:
        chapter_dir = flac_path.parent
        prefix = chapter_dir.name
        speaker_id = chapter_dir.parent.name
        trans_path = chapter_dir / f"{speaker_id}-{prefix}.trans.txt"
        if trans_path not in transcript_cache:
            mapping: Dict[str, str] = {}
            if trans_path.exists():
                for line in trans_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        utt_id, text = parts
                        mapping[utt_id.strip()] = text.strip()
            transcript_cache[trans_path] = mapping
        mapping = transcript_cache[trans_path]
        utt_stem = flac_path.stem  # e.g., 19-198-0025
        return mapping.get(utt_stem, "")

    # Convert to wav cache for chosen speaker only and capture reference transcripts
    items: List[Dict] = []
    for idx, flac_path in enumerate(tqdm(chosen_files, desc=f"Caching speaker {chosen}")):
        spk_dir = cache_root / chosen
        spk_dir.mkdir(parents=True, exist_ok=True)
        out_path = spk_dir / f"clip_{idx:05d}.wav"
        if not out_path.exists():
            data, sr = sf.read(str(flac_path))
            sf.write(str(out_path), data, int(sr))
        items.append({"path": out_path, "sentence": get_transcript_for(flac_path)})

    random.Random(seed).shuffle(items)
    n_train = max(1, int(len(items) * train_frac))
    train_items = items[:n_train]
    test_items = items[n_train:]
    if not test_items:
        raise RuntimeError("Not enough clips for a test split; increase --min-clips or adjust --train-frac.")

    return chosen, Split(
        train_paths=[x["path"] for x in train_items],
        test_paths=[x["path"] for x in test_items],
        path_to_transcript={x["path"]: x["sentence"] for x in items},
    )


def _choose_from_vctk(dataset_root: Path, min_clips: int, seed: int, train_frac: float, speaker_id: Optional[str] = None) -> Tuple[str, Split]:
    cache_root = dataset_root / "data" / "vctk_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    raw_root = dataset_root / "data" / "vctk_raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    ds = torchaudio.datasets.VCTK(
        root=str(raw_root),
        download=True,
    )

    by_speaker: Dict[str, List[Dict]] = {}
    chosen: str | None = None

    for i in tqdm(range(len(ds)), desc="Indexing VCTK"):
        waveform, sr, utterance, speaker_id, _ = ds[i]
        spk = str(speaker_id)
        spk_dir = cache_root / spk
        spk_dir.mkdir(parents=True, exist_ok=True)
        out_path = spk_dir / f"clip_{i:05d}.wav"
        if not out_path.exists():
            torchaudio.save(str(out_path), waveform, int(sr))
        by_speaker.setdefault(spk, []).append({"path": out_path, "sentence": utterance})
        if chosen is None and len(by_speaker[spk]) >= min_clips:
            chosen = spk
            break

    if chosen is None:
        if speaker_id and speaker_id in by_speaker:
            chosen = speaker_id
        else:
            if not by_speaker:
                raise RuntimeError("VCTK yielded no items.")
            chosen = max(by_speaker.keys(), key=lambda k: len(by_speaker[k]))

    items = by_speaker[chosen]
    random.Random(seed).shuffle(items)
    n_train = max(1, int(len(items) * train_frac))
    train_items = items[:n_train]
    test_items = items[n_train:]
    if not test_items:
        raise RuntimeError("Not enough clips for a test split; increase --min-clips or adjust --train-frac.")

    return chosen, Split(
        train_paths=[x["path"] for x in train_items],
        test_paths=[x["path"] for x in test_items],
    )


def choose_speaker_and_split(dataset_root: Path, min_clips: int, seed: int, train_frac: float, dataset: str, speaker_id: Optional[str] = None) -> Tuple[str, Split]:
    if dataset == "librispeech":
        return _choose_from_librispeech(dataset_root, min_clips, seed, train_frac, speaker_id)
    if dataset == "vctk":
        return _choose_from_vctk(dataset_root, min_clips, seed, train_frac, speaker_id)
    # Fallback: try streaming CV/LibriSpeech via datasets (may fail depending on hub)
    # ... existing streaming implementation remains here ...
    # Try multiple Common Voice versions in streaming mode
    dataset_candidates = [
        {
            "name": "librispeech_asr",
            "config": "clean",
            "splits": ["train.clean.100", "validation.clean", "test.clean"],
            "speaker_key": "speaker_id",
            "text_key": "text",
            "cache_subdir": "librispeech_clean",
        },
        {
            "name": "mozilla-foundation/common_voice_17_0",
            "config": "en",
            "splits": ["train", "validation", "test"],
            "speaker_key": "client_id",
            "text_key": "sentence",
            "cache_subdir": "common_voice_en",
        },
    ]
    # Prefer LibriSpeech (stable) then fall back to Common Voice
    dataset_candidates = [
        {
            "name": "librispeech_asr",
            "config": "clean",
            "splits": ["train.clean.100", "validation.clean", "test.clean"],
            "speaker_key": "speaker_id",
            "text_key": "text",
            "cache_subdir": "librispeech_clean",
        },
        {
            "name": "mozilla-foundation/common_voice_17_0",
            "config": "en",
            "splits": ["train", "validation", "test"],
            "speaker_key": "client_id",
            "text_key": "sentence",
            "cache_subdir": "common_voice_en",
        },
    ]

    last_error: Exception | None = None
    splits = []
    spec_used = None
    for spec in dataset_candidates:
        try:
            splits = [
                load_dataset(spec["name"], spec["config"], split=sp, streaming=True)
                for sp in spec["splits"]
            ]
            spec_used = spec
            break
        except Exception as e:  # noqa: BLE001
            last_error = e
            splits = []
            continue
    if not splits or spec_used is None:
        tried = [(d["name"], d["config"]) for d in dataset_candidates]
        raise RuntimeError(
            f"Failed to load streaming dataset (tried {tried}).\nLast error: {last_error}"
        )

    cache_root = dataset_root / "data" / spec_used["cache_subdir"]
    cache_root.mkdir(parents=True, exist_ok=True)

    by_speaker: Dict[str, List[Dict]] = {}
    chosen_speaker: str | None = None

    for row in itertools.chain.from_iterable(splits):
        spk = row.get(spec_used["speaker_key"]) or row.get("speaker_id") or row.get("client_id")
        if not spk:
            continue
        # Try to use existing file path if present
        path = row.get("path")
        if not path and isinstance(row.get("audio"), dict):
            path = row["audio"].get("path")

        local_path: Path | None = None
        if path:
            p = Path(path)
            if p.exists():
                local_path = p

        # If no local file, materialize audio array to cache
        if local_path is None:
            audio = row.get("audio") or {}
            arr = audio.get("array")
            sr = audio.get("sampling_rate")
            if arr is None or sr is None:
                continue
            spk_dir = cache_root / spk
            spk_dir.mkdir(parents=True, exist_ok=True)
            idx = len(by_speaker.get(spk, [])) + 1
            candidate = spk_dir / f"clip_{idx:05d}.wav"
            if not candidate.exists():
                sf.write(str(candidate), arr, int(sr))
            local_path = candidate

        text_val = row.get(spec_used["text_key"]) or row.get("text") or row.get("sentence") or ""
        by_speaker.setdefault(spk, []).append({"path": local_path, "sentence": text_val})

        if chosen_speaker is None and len(by_speaker[spk]) >= min_clips:
            chosen_speaker = spk
            break

    if chosen_speaker is None:
        # If no one reached min_clips during streaming, pick the best we have
        if not by_speaker:
            raise RuntimeError("No audio examples found in Common Voice stream.")
        chosen_speaker = max(by_speaker.keys(), key=lambda k: len(by_speaker[k]))

    items = by_speaker[chosen_speaker]
    random.Random(seed).shuffle(items)
    n_train = max(1, int(len(items) * train_frac))
    train_items = items[:n_train]
    test_items = items[n_train:]

    if not test_items:
        raise RuntimeError("Not enough clips for a test split; increase --min-clips or adjust --train-frac.")

    return chosen_speaker, Split(
        train_paths=[x["path"] for x in train_items],
        test_paths=[x["path"] for x in test_items],
    )


def run_baseline(
    clone_out_dir: Path,
    prompts_dir: Path,
    prompt_file: Path,
    min_clips: int,
    seed: int,
    train_frac: float,
    sample_rate: int,
    max_train_clones: int | None,
    dataset: str,
    eval_downsample_frac: float,
    speaker_id: Optional[str] = None,
    system_prompt: str = "DEFAULT",
    eval_count: Optional[int] = None,
) -> None:
    clone_out_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)

    speaker_id, split = choose_speaker_and_split(
        Path.cwd(), min_clips=min_clips, seed=seed, train_frac=train_frac, dataset=dataset, speaker_id=speaker_id
    )

    # load prompt text
    prompt_text = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt_text:
        prompt_text = "Please read the sentence naturally."

    generator = VoiceGenerator()
    analyzer = VoiceAnalyzer(sample_rate=sample_rate)
    embedder = SpeakerEmbedder(cache_dir=Path.cwd() / "data" / "emb_cache")
    wavlm = WavLMEmbedder(cache_dir=Path.cwd() / "data" / "wavlm_cache")
    # Lazy ASR pipeline for WER (whisper-small / faster)
    asr: Any | None = None

    # Generate clones from train set (10%)
    clone_paths: List[Path] = []
    to_clone = split.train_paths if max_train_clones is None else split.train_paths[:max_train_clones]
    for i, ref in enumerate(tqdm(to_clone, desc="Cloning")):
        # Look up transcript for this reference (if available)
        reference_transcript = split.path_to_transcript.get(ref, "")
        audio = generator.generate_impersonation(
            target_voice_path=str(ref),
            text=prompt_text,
            strategy="direct_cloning",
            reference_transcript=reference_transcript,
            system_prompt=system_prompt,
        )
        out_path = clone_out_dir / f"clone_{i:04d}.wav"
        generator.save_audio(audio, str(out_path), sample_rate=sample_rate)
        clone_paths.append(out_path)

    # Select evaluation subset: prefer a fixed eval_count if provided, else optional fraction
    eval_test_paths = split.test_paths
    if eval_count is not None and eval_count > 0 and len(eval_test_paths) > eval_count:
        rng = random.Random(seed)
        eval_test_paths = rng.sample(eval_test_paths, eval_count)
    elif 0.0 < eval_downsample_frac < 1.0 and len(eval_test_paths) > 1:
        rng = random.Random(seed)
        k = max(1, int(len(eval_test_paths) * eval_downsample_frac))
        eval_test_paths = rng.sample(eval_test_paths, k)

    # Evaluate similarities
    def mean_overall(pairs: List[Tuple[Path, Path]]) -> float:
        vals: List[float] = []
        for a, b in tqdm(pairs, desc="Scoring"):
            s = analyzer.compare_voices(str(a), str(b))
            vals.append(float(s["overall_similarity"]))
        return float(sum(vals) / max(1, len(vals)))

    def mean_embedding_cosine(pairs: List[Tuple[Path, Path]]) -> float:
        vals: List[float] = []
        for a, b in tqdm(pairs, desc="Embedding Scoring"):
            ea = embedder.embed_file(str(a))
            eb = embedder.embed_file(str(b))
            vals.append(SpeakerEmbedder.cosine_similarity(ea, eb))
        return float(sum(vals) / max(1, len(vals)))

    def mean_wavlm_cosine(pairs: List[Tuple[Path, Path]]) -> float:
        vals: List[float] = []
        for a, b in tqdm(pairs, desc="WavLM Scoring"):
            ea = wavlm.embed_file(str(a))
            eb = wavlm.embed_file(str(b))
            vals.append(WavLMEmbedder.cosine_similarity(ea, eb))
        return float(sum(vals) / max(1, len(vals)))

    def mean_wer(refs: List[Path], hyps: List[Path]) -> float:
        nonlocal asr
        if asr is None:
            asr = hf_pipeline("automatic-speech-recognition", model="openai/whisper-small", device=-1)
        ref_texts: List[str] = []
        hyp_texts: List[str] = []
        # truncate to same length for pairing
        n = min(len(refs), len(hyps))
        for i in tqdm(range(n), desc="WER ASR"):
            r = str(refs[i]); h = str(hyps[i])
            ref_texts.append(asr(r)["text"])  # ASR on reference
            hyp_texts.append(asr(h)["text"])  # ASR on hypothesis
        if not ref_texts:
            return 0.0
        return float(jiwer_wer(ref_texts, hyp_texts))

    # Real 10% vs Real 90% (upper bound)
    rr_pairs = [(a, b) for a in split.train_paths for b in eval_test_paths]
    rr_mean = mean_overall(rr_pairs)
    rr_emb = mean_embedding_cosine(rr_pairs)
    rr_wavlm = mean_wavlm_cosine(rr_pairs)

    # Clone vs Real 90% (clone performance)
    cr_pairs = [(a, b) for a in clone_paths for b in eval_test_paths]
    cr_mean = mean_overall(cr_pairs)
    cr_emb = mean_embedding_cosine(cr_pairs)
    cr_wavlm = mean_wavlm_cosine(cr_pairs)

    results = {
        "speaker_id": speaker_id,
        "num_train": len(split.train_paths),
        "num_test": len(split.test_paths),
        "num_test_evaluated": len(eval_test_paths),
        "num_clones": len(clone_paths),
        "mean_overall_real_vs_real": rr_mean,
        "mean_overall_clone_vs_real": cr_mean,
        "mean_embed_real_vs_real": rr_emb,
        "mean_embed_clone_vs_real": cr_emb,
        "mean_wavlm_real_vs_real": rr_wavlm,
        "mean_wavlm_clone_vs_real": cr_wavlm,
        "prompt_file": str(prompt_file),
        "system_prompt": system_prompt,
    }

    (clone_out_dir / "baseline_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(json.dumps(results, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Common Voice baseline cloning and evaluation")
    parser.add_argument("--min-clips", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.10)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-train-clones", type=int, default=5)
    parser.add_argument("--dataset", type=str, choices=["librispeech", "vctk", "auto"], default="librispeech")
    parser.add_argument("--eval-downsample-frac", type=float, default=1.0,
                        help="Fraction of test set to evaluate (e.g., 0.1 for 10%)")
    parser.add_argument("--speaker-id", type=str, default=None,
                        help="Optional fixed speaker id (e.g., 211 for LibriSpeech)")
    parser.add_argument("--system-prompt", type=str, default="DEFAULT",
                        help="System prompt to use (or DEFAULT to use built-in)")
    parser.add_argument("--eval-count", type=int, default=None,
                        help="If set, evaluate exactly this many test items (overrides fraction)")

    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--clone-dir", type=Path, default=project_root / "src" / "clone-audio")
    parser.add_argument("--prompts-dir", type=Path, default=project_root / "src" / "prompts")
    parser.add_argument("--prompt-file", type=Path, default=project_root / "src" / "prompts" / "baseline_prompt_1.txt")
    parser.add_argument("--prompt-files", type=Path, nargs="+", default=None,
                        help="Optional list of prompt files to evaluate separately")

    args = parser.parse_args()

    if args.prompt_files:
        base_clone_dir = args.clone_dir
        for pf in args.prompt_files:
            run_baseline(
                clone_out_dir=base_clone_dir / pf.stem,
                prompts_dir=args.prompts_dir,
                prompt_file=pf,
                min_clips=args.min_clips,
                seed=args.seed,
                train_frac=args.train_frac,
                sample_rate=args.sample_rate,
                max_train_clones=args.max_train_clones,
                dataset=("librispeech" if args.dataset == "auto" else args.dataset),
                eval_downsample_frac=args.eval_downsample_frac,
                speaker_id=args.speaker_id,
                system_prompt=args.system_prompt,
                eval_count=args.eval_count,
            )
    else:
        run_baseline(
            clone_out_dir=args.clone_dir,
            prompts_dir=args.prompts_dir,
            prompt_file=args.prompt_file,
            min_clips=args.min_clips,
            seed=args.seed,
            train_frac=args.train_frac,
            sample_rate=args.sample_rate,
            max_train_clones=args.max_train_clones,
            dataset=("librispeech" if args.dataset == "auto" else args.dataset),
            eval_downsample_frac=args.eval_downsample_frac,
            speaker_id=args.speaker_id,
            system_prompt=args.system_prompt,
            eval_count=args.eval_count,
        )


if __name__ == "__main__":
    main()


