from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Dict, Any, Tuple
import random

import soundfile as sf

from src.voice_generator import VoiceGenerator
from experiments.full_eval.data_prep import normalize_audio, save_wav, RANDOM_SEED
from src.embedding_scorer import SpeakerEmbedder, ecapa_cosine
from experiments.full_eval.metrics import (
    MetricCaches,
    WavLMEmbedder,
    wavlm_cosine,
    mfcc_cosine,
    pitch_similarity,
)
from src.voice_analyzer import VoiceAnalyzer
from experiments.full_eval.pairing import build_pairs_rr, build_pairs_cr, build_pairs_ci
from sklearn.metrics import roc_auc_score, roc_curve
from jiwer import wer as jiwer_wer


# Keep sentences aligned with experiments/full_eval/run_full_experiment.py
FIXED_SENTENCES = [
    "Hello, my name is Alex, and I love playing the piano.",
    "It’s a beautiful day to learn something new.",
    "I believe creativity is the heart of innovation.",
    "Let’s take a deep breath and start fresh.",
    "The future belongs to those who prepare for it today.",
    "I often walk by the river to clear my mind.",
    "Please make sure you turn off the lights before leaving.",
    "Good morning, and welcome to our daily briefing.",
    "I can’t believe how quickly this year has passed.",
    "Teamwork makes everything possible.",
]


def find_speaker_flacs(extracted_root: Path, speaker_id: str) -> List[Path]:
    spk_dir = extracted_root / speaker_id
    return sorted(spk_dir.rglob("*.flac"))


def pick_references(flacs: List[Path]) -> List[Path]:
    # pick 2–3 clips, each 6–12s, total 15–30s; skip too short
    rng = random.Random(RANDOM_SEED)
    candidates: List[Tuple[Path, float]] = []
    for f in flacs:
        try:
            info = sf.info(str(f))
            dur = float(info.frames) / float(info.samplerate)
            if 6.0 <= dur <= 12.0:
                candidates.append((f, dur))
        except Exception:
            continue
    rng.shuffle(candidates)
    best: List[Path] = []
    total = 0.0
    for f, dur in candidates:
        if len(best) < 3 and total + dur <= 30.0:
            best.append(f)
            total += dur
        if len(best) >= 2 and 15.0 <= total <= 30.0:
            break
    if len(best) < 2 and candidates:
        best = [candidates[0][0]]
    return best[:3]


def decoding_grids(kind: str) -> List[Dict[str, Any]]:
    # Provide two grids to choose from
    if kind == "small":
        temps = [0.7, 1.0]
        top_ps = [0.9, 0.95]
        top_ks = [20, 50]
    else:  # "large"
        temps = [0.5, 0.7, 1.0, 1.2]
        top_ps = [0.85, 0.9, 0.95, 0.98]
        top_ks = [10, 20, 50, 100]

    combos: List[Dict[str, Any]] = []
    for t in temps:
        for p in top_ps:
            for k in top_ks:
                combos.append({"temperature": t, "top_p": p, "top_k": k})
    return combos


def main() -> None:
    parser = argparse.ArgumentParser(description="Decoding grid search using leaderboard prompts")
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "experiments" / "data"
    extracted_tc = data_root / "librispeech_extracted" / "LibriSpeech" / "test-clean"
    extracted_tr = data_root / "librispeech_extracted" / "LibriSpeech" / "train-clean-100"
    extracted = extracted_tc if extracted_tc.exists() else extracted_tr

    parser.add_argument("--leaderboard_csv", type=Path, default=project_root / "experiments" / "results" / "prompt_leaderboard_top5_per_speaker.csv")
    parser.add_argument("--grid", type=str, choices=["small", "large"], default="small")
    parser.add_argument("--out_csv", type=Path, default=project_root / "experiments" / "results" / "decoding_grid" / "grid_results.csv")
    parser.add_argument("--out_wavs", type=Path, default=project_root / "experiments" / "outputs_decoding_grid")
    args = parser.parse_args()

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_wavs.mkdir(parents=True, exist_ok=True)

    # Load leaderboard selections
    rows: List[Dict[str, str]] = []
    with args.leaderboard_csv.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            # skip rows without prompt_style_id/sentence_id
            if not r.get("speaker_id"):
                continue
            if not r.get("prompt_style_id") or not r.get("sentence_id"):
                continue
            rows.append(r)

    # Group by speaker_id
    by_speaker: Dict[str, List[Dict[str, str]]] = {}
    for r in rows:
        by_speaker.setdefault(str(r["speaker_id"]), []).append(r)

    # For each speaker, keep only the setting with highest cr_rr_ratio
    best_by_speaker: Dict[str, List[Dict[str, str]]] = {}
    for spk, items in by_speaker.items():
        best_row: Dict[str, str] | None = None
        best_score: float | None = None
        for r in items:
            raw = r.get("cr_rr_ratio", "nan")
            try:
                score = float(raw)
            except Exception:
                score = float("nan")
            # skip NaN
            if score != score:
                continue
            if best_score is None or score > best_score:
                best_score = score
                best_row = r
        if best_row is None and items:
            best_row = items[0]
        if best_row is not None:
            best_by_speaker[spk] = [best_row]
    by_speaker = best_by_speaker

    # Initialize engines
    vg = VoiceGenerator()
    embedder = SpeakerEmbedder(cache_dir=project_root / "cache" / "ecapa")
    caches = MetricCaches(project_root / "cache")
    wavlm = WavLMEmbedder(cache_dir=caches.emb_cache)
    analyzer = VoiceAnalyzer(sample_rate=16000)

    header = [
        "speaker_id", "prompt_style_id", "sentence_id",
        "temperature", "top_p", "top_k",
        # Aggregates matching previous experiment
        "cr_mean_wavlm", "rr_mean_wavlm", "ci_mean_wavlm",
        "cr_rr_ratio", "auc_wavlm", "eer_wavlm", "cr_median_wer",
        "count_cr", "count_rr", "count_ci",
        # Additional metrics
        "ecapa_cr_mean", "ecapa_rr_mean", "ecapa_ci_mean",
        # bookkeeping
        "clone_path", "ref_paths", "real_paths",
    ]
    write_header = not args.out_csv.exists()
    with args.out_csv.open("a", newline="", encoding="utf-8") as fcsv:
        wr = csv.writer(fcsv)
        if write_header:
            wr.writerow(header)

        for spk, items in by_speaker.items():
            print(f"[Grid] Speaker {spk}: preparing references and reals")
            flacs = find_speaker_flacs(extracted, spk)
            refs_flac = pick_references(flacs)

            # materialize references as wav @16k
            ref_wavs: List[Path] = []
            for i, rf in enumerate(refs_flac):
                try:
                    y, sr = sf.read(str(rf))
                except Exception:
                    continue
                y, sr = normalize_audio(y, sr, target_sr=16000, target_lufs=-23.0)
                out_ref = args.out_wavs / spk / "refs" / f"ref_{i:02d}.wav"
                save_wav(out_ref, y, sr)
                ref_wavs.append(out_ref)

            # build 10 real evaluation wavs
            real_wavs: List[Path] = []
            rng = random.Random(RANDOM_SEED)
            real_candidates = list(flacs)
            rng.shuffle(real_candidates)
            for fl in real_candidates:
                if len(real_wavs) >= 10:
                    break
                try:
                    y, sr = sf.read(str(fl))
                except Exception:
                    continue
                y, sr = normalize_audio(y, sr, target_sr=16000, target_lufs=-23.0)
                out_real = args.out_wavs / spk / "reals" / f"real_{len(real_wavs):02d}.wav"
                save_wav(out_real, y, sr)
                real_wavs.append(out_real)

            grids = decoding_grids(args.grid)
            for r in items:
                style_id = str(r["prompt_style_id"]).strip()
                sent_id = int(float(r["sentence_id"]))  # handle CSV numeric strings
                sentence = FIXED_SENTENCES[sent_id - 1] if 1 <= sent_id <= len(FIXED_SENTENCES) else FIXED_SENTENCES[0]
                # cycle reference
                ref_idx = 0

                for g in grids:
                    text = f"[SPEAKER1] {sentence}"
                    ref_path = ref_wavs[ref_idx % max(1, len(ref_wavs))] if ref_wavs else None
                    ref_idx += 1
                    if ref_path is None:
                        print(f"[Grid] Speaker {spk}: no references available, skipping")
                        continue

                    audio_bytes = vg.generate_impersonation(
                        target_voice_path=str(ref_path),
                        text=text,
                        strategy="direct_cloning",
                        reference_transcript="",
                        system_prompt=(
                            "You are an AI assistant designed to convert text into speech.\n"
                            "If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.\n"
                            "If no speaker tag is present, select a suitable voice on your own.\n"
                            "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>\n\n"
                            "CRITICAL: Preserve the speaker's identity; avoid changing their natural pitch, timbre, or style."
                        ),
                        temperature=float(g["temperature"]),
                        top_p=float(g["top_p"]),
                        top_k=int(g["top_k"]),
                    )

                    out_clone = args.out_wavs / spk / "clones" / f"{style_id}_s{sent_id}_t{g['temperature']}_p{g['top_p']}_k{g['top_k']}.wav"
                    out_clone.parent.mkdir(parents=True, exist_ok=True)
                    vg.save_audio(audio_bytes, str(out_clone), sample_rate=24000)

                    # Build RR pairs from real_wavs, CR/CI pairs using this single clone
                    rr_pairs = build_pairs_rr(real_wavs, k_pairs=20)
                    cr_pairs = build_pairs_cr([out_clone], real_wavs, per_clone=3)
                    # Build impostors by sampling other speakers
                    impostors: List[Path] = []
                    # find up to 10 impostor wavs from other speaker dirs
                    other_dirs = [d for d in extracted.iterdir() if d.is_dir() and d.name != spk]
                    rng = random.Random(RANDOM_SEED)
                    rng.shuffle(other_dirs)
                    for od in other_dirs:
                        od_flacs = sorted(od.rglob("*.flac"))
                        rng.shuffle(od_flacs)
                        for fl in od_flacs[:2]:
                            try:
                                y, sr = sf.read(str(fl))
                            except Exception:
                                continue
                            y, sr = normalize_audio(y, sr, target_sr=16000, target_lufs=-23.0)
                            out_imp = args.out_wavs / spk / "impostors" / f"{od.name}_imp_{len(impostors):02d}.wav"
                            save_wav(out_imp, y, sr)
                            impostors.append(out_imp)
                            if len(impostors) >= 10:
                                break
                        if len(impostors) >= 10:
                            break
                    ci_pairs = build_pairs_ci([out_clone], impostors, per_clone=min(10, len(impostors)))

                    # Collect WavLM similarities per split
                    rr_wavlm: List[float] = []
                    cr_wavlm: List[float] = []
                    ci_wavlm: List[float] = []
                    # Also ECAPA means per split
                    rr_ecapa: List[float] = []
                    cr_ecapa: List[float] = []
                    ci_ecapa: List[float] = []

                    for a, b in rr_pairs:
                        try:
                            rr_wavlm.append(float(wavlm_cosine(a, b, wavlm)))
                            rr_ecapa.append(float(ecapa_cosine(a, b, embedder)))
                        except Exception:
                            continue
                    for a, b in cr_pairs:
                        try:
                            cr_wavlm.append(float(wavlm_cosine(a, b, wavlm)))
                            cr_ecapa.append(float(ecapa_cosine(a, b, embedder)))
                        except Exception:
                            continue
                    for a, b in ci_pairs:
                        try:
                            ci_wavlm.append(float(wavlm_cosine(a, b, wavlm)))
                            ci_ecapa.append(float(ecapa_cosine(a, b, embedder)))
                        except Exception:
                            continue

                    def mean_or_zero(xs: List[float]) -> float:
                        return float(sum(xs) / len(xs)) if xs else 0.0

                    rr_mean = mean_or_zero(rr_wavlm)
                    cr_mean = mean_or_zero(cr_wavlm)
                    ci_mean = mean_or_zero(ci_wavlm)
                    cr_rr_ratio = (cr_mean / rr_mean) if rr_mean > 0 else 0.0

                    # AUC/EER on WavLM (CR positives, CI negatives)
                    auc = 0.0
                    eer = 0.0
                    y_true: List[int] = [1] * len(cr_wavlm) + [0] * len(ci_wavlm)
                    y_score: List[float] = cr_wavlm + ci_wavlm
                    if len(set(y_true)) == 2 and len(y_score) >= 2:
                        try:
                            auc = float(roc_auc_score(y_true, y_score))
                            fpr, tpr, thr = roc_curve(y_true, y_score)
                            # EER where FPR ~= 1-TPR
                            fnr = 1 - tpr
                            idx = int(min(range(len(fpr)), key=lambda i: abs(fpr[i] - fnr[i])))
                            eer = float((fpr[idx] + fnr[idx]) / 2.0)
                        except Exception:
                            pass

                    # CR median WER: transcribe each CR pair and compute per-pair WER
                    cr_wer_vals: List[float] = []
                    try:
                        from experiments.full_eval.metrics import ASREngine
                        asr = ASREngine()
                        for a, b in cr_pairs:
                            try:
                                ref_txt = asr.transcribe(b)
                                hyp_txt = asr.transcribe(a)
                                cr_wer_vals.append(float(jiwer_wer(ref_txt, hyp_txt)))
                            except Exception:
                                continue
                    except Exception:
                        pass
                    cr_median_wer = float(sorted(cr_wer_vals)[len(cr_wer_vals)//2]) if cr_wer_vals else 0.0

                    # ECAPA per split means
                    ecapa_rr_mean = mean_or_zero(rr_ecapa)
                    ecapa_cr_mean = mean_or_zero(cr_ecapa)
                    ecapa_ci_mean = mean_or_zero(ci_ecapa)

                    wr.writerow([
                        spk, style_id, sent_id,
                        g["temperature"], g["top_p"], g["top_k"],
                        cr_mean, rr_mean, ci_mean,
                        cr_rr_ratio, auc, eer, cr_median_wer,
                        len(cr_pairs), len(rr_pairs), len(ci_pairs),
                        ecapa_cr_mean, ecapa_rr_mean, ecapa_ci_mean,
                        str(out_clone),
                        ";".join(str(p) for p in ref_wavs), ";".join(str(p) for p in real_wavs),
                    ])


if __name__ == "__main__":
    main()


