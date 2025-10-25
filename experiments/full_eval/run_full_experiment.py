from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import soundfile as sf

from src.voice_generator import VoiceGenerator
from experiments.full_eval.data_prep import pick_top_speakers, normalize_audio, save_wav, RANDOM_SEED
from experiments.full_eval.metrics import MetricCaches, WavLMEmbedder, mfcc_cosine, pitch_similarity, wavlm_cosine
from experiments.full_eval.pairing import build_pairs_rr, build_pairs_cr, build_pairs_ci
from src.voice_analyzer import VoiceAnalyzer


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

P1 = "Neutral"
P2 = "Identity-Preserving"
SYSTEM_PROMPT = (
    "You are an AI assistant designed to convert text into speech.\n"
    "If the user's message includes a [SPEAKER*] tag, do not read out the tag and generate speech for the following text, using the specified voice.\n"
    "If no speaker tag is present, select a suitable voice on your own.\n\n"
    "<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>\n\n"
    "CRITICAL: Preserve the speaker's identity; avoid changing their natural pitch, timbre, or style."
)


def load_transcript_for_flac(flac_path: Path) -> str:
    chapter_dir = flac_path.parent
    speaker_id = chapter_dir.parent.name
    trans_path = chapter_dir / f"{speaker_id}-{chapter_dir.name}.trans.txt"
    if not trans_path.exists():
        return ""
    stem = flac_path.stem  # e.g., 19-198-0025
    for line in trans_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split(" ", 1)
        if len(parts) == 2 and parts[0] == stem:
            return parts[1].strip()
    return ""


def find_speaker_flacs(extracted_root: Path, speaker_id: str) -> List[Path]:
    spk_dir = extracted_root / speaker_id
    return sorted(spk_dir.rglob("*.flac"))


def pick_references(flacs: List[Path]) -> List[Tuple[Path, str]]:
    # pick 2–3 clips, each 6–12s, total 15–30s; skip too short
    rng = random.Random(RANDOM_SEED)
    candidates: List[Tuple[Path, float, str]] = []
    for f in flacs:
        try:
            info = sf.info(str(f))
            dur = float(info.frames) / float(info.samplerate)
            if 6.0 <= dur <= 12.0:
                candidates.append((f, dur, load_transcript_for_flac(f)))
        except Exception:
            continue
    rng.shuffle(candidates)
    best: List[Tuple[Path, str]] = []
    total = 0.0
    for f, dur, tr in candidates:
        if len(best) < 3 and total + dur <= 30.0:
            best.append((f, tr))
            total += dur
        if len(best) >= 2 and 15.0 <= total <= 30.0:
            break
    if len(best) < 2 and candidates:
        best = [(candidates[0][0], candidates[0][2])]
    return best[:3]


def main() -> None:
    parser = argparse.ArgumentParser(description="Full voice cloning evaluation experiment")
    project_root = Path(__file__).resolve().parents[2]
    data_root = project_root / "experiments" / "data"
    # Prefer test-clean, fallback to train-clean-100
    extracted_tc = data_root / "librispeech_extracted" / "LibriSpeech" / "test-clean"
    extracted_tr = data_root / "librispeech_extracted" / "LibriSpeech" / "train-clean-100"
    extracted = extracted_tc if extracted_tc.exists() else extracted_tr
    out_csv = project_root / "experiments" / "results" / "results.csv"
    out_wavs = project_root / "experiments" / "outputs"

    parser.add_argument("--speakers", type=int, default=3)
    parser.add_argument("--api_concurrency", type=int, default=2)
    args = parser.parse_args()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_wavs.mkdir(parents=True, exist_ok=True)
    print(f"[FullEval] Using extracted data at: {extracted}")
    print(f"[FullEval] Writing WAVs to: {out_wavs}")

    # Select speakers (top-N by clip count, ties by smaller ID)
    speakers = pick_top_speakers(extracted, top_n=args.speakers)
    print(f"[FullEval] Selected speakers: {speakers}")

    # Initialize generator and metrics
    vg = VoiceGenerator()
    caches = MetricCaches(project_root / "cache")
    wavlm = WavLMEmbedder(cache_dir=caches.emb_cache)
    analyzer = VoiceAnalyzer(sample_rate=16000)

    header = [
        "speaker_id", "pair_type", "clone_id", "real_id", "impostor_speaker_id",
        "system_id", "prompt_style_id", "sentence_id", "temperature", "top_p", "top_k",
        "wavlm_sim", "mfcc_sim", "pitch_sim", "acoustic_metric", "wer", "clone_path", "real_path", "timestamp",
    ]
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        if write_header:
            wr.writerow(header)

        for spk in speakers:
            print(f"[FullEval] Speaker {spk}: preparing files")
            spk_flacs = find_speaker_flacs(extracted, spk)
            refs = pick_references(spk_flacs)
            print(f"[FullEval] Speaker {spk}: picked {len(refs)} references")
            ref_wavs: List[Path] = []
            ref_trs: List[str] = []
            # Normalize and materialize references as wav
            for i, (rf, tr) in enumerate(refs):
                y, sr = sf.read(str(rf))
                y, sr = normalize_audio(y, sr, target_sr=16000, target_lufs=-23.0)
                out_ref = out_wavs / spk / "refs" / f"ref_{i:02d}.wav"
                save_wav(out_ref, y, sr)
                ref_wavs.append(out_ref)
                ref_trs.append(tr)

            # Choose 10 real evaluation audios (non-overlapping)
            reals: List[Path] = []
            seen = set(p for p in ref_wavs)
            rng = random.Random(RANDOM_SEED)
            real_candidates = [p for p in spk_flacs]
            rng.shuffle(real_candidates)
            print(f"[FullEval] Speaker {spk}: building 10 reals")
            for fl in real_candidates:
                if len(reals) >= 10:
                    break
                try:
                    y, sr = sf.read(str(fl))
                except Exception:
                    continue
                y, sr = normalize_audio(y, sr, target_sr=16000, target_lufs=-23.0)
                out_real = out_wavs / spk / "reals" / f"real_{len(reals):02d}.wav"
                save_wav(out_real, y, sr)
                reals.append(out_real)

            # Impostors: 10 clips from other speakers
            impostor_paths: List[Path] = []
            other_spk = [s for s in speakers if s != spk]
            for osid in other_spk:
                os_flacs = find_speaker_flacs(extracted, osid)
                rng.shuffle(os_flacs)
                for fl in os_flacs[:3]:
                    try:
                        y, sr = sf.read(str(fl))
                    except Exception:
                        continue
                    y, sr = normalize_audio(y, sr, target_sr=16000, target_lufs=-23.0)
                    out_imp = out_wavs / spk / "impostors" / f"{osid}_imp_{len(impostor_paths):02d}.wav"
                    save_wav(out_imp, y, sr)
                    impostor_paths.append(out_imp)
                    if len(impostor_paths) >= 10:
                        break
                if len(impostor_paths) >= 10:
                    break

            # Generate 10 clones: 5 per style, fixed sentences
            print(f"[FullEval] Speaker {spk}: generating 10 clones")
            clones: List[Tuple[Path, str, str, int]] = []  # path, style_id, ref_speaker_tag, sentence_id
            for idx, sentence in enumerate(FIXED_SENTENCES):
                style = P1 if idx < 5 else P2
                # cycle through references
                r_index = idx % max(1, len(ref_wavs))
                ref_path = ref_wavs[r_index]
                ref_trans = ref_trs[r_index] if r_index < len(ref_trs) else ""
                # include speaker tag
                user_text = f"[SPEAKER1] {sentence}"
                audio_bytes = vg.generate_impersonation(
                    target_voice_path=str(ref_path),
                    text=user_text,
                    strategy="direct_cloning",
                    reference_transcript=ref_trans,
                    system_prompt=SYSTEM_PROMPT,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50,
                )
                out_clone = out_wavs / spk / "clones" / f"clone_{idx:02d}.wav"
                out_clone.parent.mkdir(parents=True, exist_ok=True)
                vg.save_audio(audio_bytes, str(out_clone), sample_rate=24000)
                # post-normalize clone
                try:
                    y, sr = sf.read(str(out_clone))
                    y, sr = normalize_audio(y, sr, target_sr=16000, target_lufs=-23.0)
                    save_wav(out_clone, y, sr)
                except Exception:
                    pass
                clones.append((out_clone, style, "S1", idx + 1))

            # Build pairs per spec
            print(f"[FullEval] Speaker {spk}: pairing RR/CR/CI")
            rr_pairs = build_pairs_rr(reals, k_pairs=20)
            cr_pairs = build_pairs_cr([c[0] for c in clones], reals, per_clone=3)
            ci_pairs = build_pairs_ci([c[0] for c in clones], impostor_paths, per_clone=10)

            # Score and write rows
            print(f"[FullEval] Speaker {spk}: scoring RR pairs ({len(rr_pairs)})")
            for a, b in rr_pairs:
                wr.writerow([
                    spk, "RR", "", "", "",
                    "S1", "", "", 1.0, 0.95, 50,
                    wavlm_cosine(a, b, wavlm), mfcc_cosine(a, b, caches), pitch_similarity(a, b, caches),
                    analyzer.compare_voices(str(a), str(b))["overall_similarity"], "",
                    str(a), str(b), datetime.utcnow().isoformat(),
                ])
            print(f"[FullEval] Speaker {spk}: scoring CR/CI pairs")
            for idx, (cpath, style, sys_id, sent_id) in enumerate(clones):
                # For CR
                targets = [p for (cp, p) in cr_pairs if cp == cpath]
                for ridx, r in enumerate(targets):
                    wr.writerow([
                        spk, "CR", f"C{idx+1}", f"R{ridx+1}", "",
                        sys_id, ("P1" if style == P1 else "P2"), sent_id, 1.0, 0.95, 50,
                        wavlm_cosine(cpath, r, wavlm), mfcc_cosine(cpath, r, caches), pitch_similarity(cpath, r, caches),
                        analyzer.compare_voices(str(cpath), str(r))["overall_similarity"], "",
                        str(cpath), str(r), datetime.utcnow().isoformat(),
                    ])
                # For CI
                imposts = [p for (cp, p) in ci_pairs if cp == cpath]
                for iid, imp in enumerate(imposts):
                    wr.writerow([
                        spk, "CI", f"C{idx+1}", "", imp.parent.name.split("_")[0] if "_" in imp.name else "",
                        sys_id, ("P1" if style == P1 else "P2"), sent_id, 1.0, 0.95, 50,
                        wavlm_cosine(cpath, imp, wavlm), mfcc_cosine(cpath, imp, caches), pitch_similarity(cpath, imp, caches),
                        analyzer.compare_voices(str(cpath), str(imp))["overall_similarity"], "",
                        str(cpath), str(imp), datetime.utcnow().isoformat(),
                    ])

    print(f"[FullEval] Wrote results to {out_csv}")


if __name__ == "__main__":
    main()


