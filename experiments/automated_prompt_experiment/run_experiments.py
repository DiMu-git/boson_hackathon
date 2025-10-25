from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import openai

from experiments.automated_prompt_experiment.higgs_eval import synthesize_audio
from src.core.embedding_scorer import SpeakerEmbedder
from tqdm import tqdm


def load_config(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(paths: List[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Higgs Audio experiments over prompts and config grids")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--config", type=Path, default=project_root / "experiment_config.json")
    parser.add_argument("--prompts-dir", type=Path, default=project_root / "src" / "prompts" / "auto")
    parser.add_argument("--out-dir", type=Path, default=project_root / "outputs")
    parser.add_argument("--csv", type=Path, default=project_root / "experiments" / "higgs_prompt_eval.csv")
    parser.add_argument("--block", type=str, default="baseline")

    args = parser.parse_args()

    ensure_dirs([args.out_dir, args.csv.parent])
    cfg = load_config(args.config)

    api_key = os.getenv("BOSON_API_KEY")
    base_url = os.getenv("BOSON_BASE_URL", "https://hackathon.boson.ai/v1")
    if not api_key:
        raise RuntimeError("BOSON_API_KEY not set")
    client = openai.Client(api_key=api_key, base_url=base_url)

    # Prepare embedder for ECAPA similarity
    embedder = SpeakerEmbedder(cache_dir=project_root / "data" / "emb_cache")

    # Baseline block: S1 + D1 + T1 + R1 + K3
    system = cfg["systems"]["S1"]
    scene = cfg["scenes"]["D1"]
    tag = cfg["tags"]["T1"]
    refs = cfg["references"]["R1"]
    decoding = cfg["decoding"]["K3"]

    header = [
        "timestamp", "block", "system_key", "scene_key", "tag_key", "decoding_key",
        "sim_ecapa", "wer", "latency_s", "content", "out_wav"
    ]
    write_header = not args.csv.exists()
    with args.csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        prompt_list = sorted((args.prompts_dir).glob("text_*.txt"))
        for idx, p in enumerate(tqdm(prompt_list, desc="Synthesizing + Scoring", unit="prompt"), start=1):
            content_text = p.read_text(encoding="utf-8").strip()

            print(f"[{idx}/{len(prompt_list)}] Synthesizing {p.name} ...", flush=True)
            resp = synthesize_audio(
                client=client,
                system_text=system,
                scene_text=scene,
                ref_paths=refs,
                speaker_tag=tag,
                content_text=content_text,
                decoding=decoding,
            )
            audio_bytes = resp["audio_bytes"]

            # Save wav
            out_wav = args.out_dir / args.block / f"{p.stem}.wav"
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            out_wav.write_bytes(audio_bytes)
            print(f"Saved audio -> {out_wav}", flush=True)

            # Compute ECAPA similarity vs first reference (simple baseline)
            emb_out = embedder.embed_file(str(out_wav))
            emb_ref = embedder.embed_file(str(Path(refs[0]).resolve()))
            sim = SpeakerEmbedder.cosine_similarity(emb_out, emb_ref)
            print(f"ECAPA cosine similarity vs ref: {sim:.4f}", flush=True)

            # Placeholder for latency/wer (not implemented here)
            latency_s = ""
            wer = ""

            writer.writerow([
                datetime.utcnow().isoformat(), args.block, "S1", "D1", "T1", "K3",
                sim, wer, latency_s, content_text, str(out_wav)
            ])

    print(f"Wrote CSV rows to {args.csv}")


if __name__ == "__main__":
    main()


