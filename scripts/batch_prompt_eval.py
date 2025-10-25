from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path
from typing import List

from src.core.common_voice_baseline import run_baseline, choose_speaker_and_split
from tqdm import tqdm
import json
import glob


def evaluate_once_for_prompt(
    prompt_file: Path,
    out_dir: Path,
    dataset: str,
    speaker_id: str | None,
    min_clips: int,
    seed: int,
    train_frac: float,
    sample_rate: int,
    max_train_clones: int | None,
    eval_downsample_frac: float,
) -> dict:
    # Reuse the existing run_baseline by directing outputs to a prompt-specific folder
    run_baseline(
        clone_out_dir=out_dir / prompt_file.stem,
        prompts_dir=out_dir.parent / "../prompts",  # not used directly inside run_baseline
        prompt_file=prompt_file,
        min_clips=min_clips,
        seed=seed,
        train_frac=train_frac,
        sample_rate=sample_rate,
        max_train_clones=max_train_clones,
        dataset=("librispeech" if dataset == "auto" else dataset),
        eval_downsample_frac=eval_downsample_frac,
        speaker_id=speaker_id,
    )
    # Read the results that run_baseline writes
    results_path = out_dir / prompt_file.stem / "baseline_results.json"
    results_dict = json.loads(results_path.read_text(encoding="utf-8"))
    results_dict["results_path"] = str(results_path)
    return results_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch evaluate prompts and append results to CSV")
    project_root = Path(__file__).resolve().parents[1]
    parser.add_argument("--prompts-glob", type=str, default=str(project_root / "src" / "prompts" / "auto" / "prompt_*.txt"))
    parser.add_argument("--out-csv", type=Path, default=project_root / "experiments" / "prompt_sweep" / "results.csv")

    parser.add_argument("--dataset", type=str, choices=["librispeech", "vctk", "auto"], default="librispeech")
    parser.add_argument("--speaker-id", type=str, default=None)
    parser.add_argument("--min-clips", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-frac", type=float, default=0.10)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--max-train-clones", type=int, default=3)
    parser.add_argument("--eval-downsample-frac", type=float, default=0.10)

    args = parser.parse_args()

    out_csv: Path = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Discover prompts
    # Support absolute patterns using glob module
    prompt_files: List[Path] = [Path(p) for p in sorted(glob.glob(args.prompts_glob))]
    if not prompt_files:
        raise FileNotFoundError(f"No prompts matched: {args.prompts_glob}")

    # CSV header
    header = [
        "timestamp",
        "speaker_id",
        "mean_overall_real_vs_real",
        "mean_overall_clone_vs_real",
        "mean_embed_real_vs_real",
        "mean_embed_clone_vs_real",
        "num_test",
        "num_test_evaluated",
        "num_clones",
        "prompt_file",
        "prompt_text",
    ]
    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)

        for i, pf in enumerate(tqdm(prompt_files, desc="Evaluating prompts", unit="prompt"), start=1):
            # Run one evaluation; results are written to JSON by run_baseline
            results = evaluate_once_for_prompt(
                prompt_file=pf,
                out_dir=project_root / "src" / "clone-audio",
                dataset=args.dataset,
                min_clips=args.min_clips,
                seed=args.seed,
                train_frac=args.train_frac,
                sample_rate=args.sample_rate,
                max_train_clones=args.max_train_clones,
                eval_downsample_frac=args.eval_downsample_frac,
                speaker_id=args.speaker_id,
            )
            # Load prompt text
            prompt_text = pf.read_text(encoding="utf-8").strip()
            print(f"[{i}/{len(prompt_files)}] {pf.name} -> speaker {results.get('speaker_id','')}, embed {results.get('mean_embed_clone_vs_real','')}")

            writer.writerow([
                datetime.utcnow().isoformat(),
                results.get("speaker_id", ""),
                results.get("mean_overall_real_vs_real", ""),
                results.get("mean_overall_clone_vs_real", ""),
                results.get("mean_embed_real_vs_real", ""),
                results.get("mean_embed_clone_vs_real", ""),
                results.get("num_test", ""),
                results.get("num_test_evaluated", results.get("num_test", "")),
                results.get("num_clones", ""),
                str(pf),
                prompt_text,
            ])

    print(f"Wrote results to {out_csv}")


if __name__ == "__main__":
    main()


