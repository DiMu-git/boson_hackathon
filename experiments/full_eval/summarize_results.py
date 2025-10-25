from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def load_rows(csv_path: Path) -> List[Dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        return [r for r in rd]


def to_float(v: str) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def group_values(rows: List[Dict[str, str]], pair_type: str, key: str) -> List[float]:
    return [to_float(r[key]) for r in rows if r.get("pair_type") == pair_type]


def summarize(values: List[float]) -> Dict[str, float]:
    arr = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "count": 0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "count": int(arr.size),
    }


def compute_auc_eer(scores_pos: List[float], scores_neg: List[float]) -> Tuple[float, float]:
    y_true = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
    y_scores = np.array(scores_pos + scores_neg)
    if len(np.unique(y_true)) < 2 or len(y_true) == 0:
        return float("nan"), float("nan")
    auc = float(roc_auc_score(y_true, y_scores))
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # EER where FNR approximately equals FPR
    idx = np.argmin(np.abs(fnr - fpr))
    eer = float((fnr[idx] + fpr[idx]) / 2)
    return auc, eer


def plot_boxplots(rows: List[Dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "RR": group_values(rows, "RR", "wavlm_sim"),
        "CR": group_values(rows, "CR", "wavlm_sim"),
        "CI": group_values(rows, "CI", "wavlm_sim"),
    }
    plt.figure(figsize=(7, 5))
    plt.boxplot([data["RR"], data["CR"], data["CI"]], labels=["RR", "CR", "CI"]) 
    plt.ylabel("WavLM similarity")
    plt.title("Similarity distributions (WavLM)")
    plt.tight_layout()
    plt.savefig(out_dir / "similarity_boxplots.png", dpi=150)
    plt.close()


def plot_roc(scores_pos: List[float], scores_neg: List[float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.array([1] * len(scores_pos) + [0] * len(scores_neg))
    y_scores = np.array(scores_pos + scores_neg)
    if len(np.unique(y_true)) < 2:
        return
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve (WavLM) - CR vs CI")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "roc_curve_wavlm.png", dpi=150)
    plt.close()


def plot_scatter_sim_wer(rows: List[Dict[str, str]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    cr_rows = [r for r in rows if r.get("pair_type") == "CR"]
    x = [to_float(r["wavlm_sim"]) for r in cr_rows]
    y = [to_float(r.get("wer", "nan")) for r in cr_rows]
    xf = np.array([xi for xi, yi in zip(x, y) if np.isfinite(xi) and np.isfinite(yi)])
    yf = np.array([yi for xi, yi in zip(x, y) if np.isfinite(xi) and np.isfinite(yi)])
    if xf.size == 0:
        return
    plt.figure(figsize=(6, 5))
    plt.scatter(xf, yf, s=12, alpha=0.6)
    plt.xlabel("WavLM similarity (CR)")
    plt.ylabel("WER (CR)")
    plt.title("Similarity vs WER (CR)")
    plt.tight_layout()
    plt.savefig(out_dir / "sim_vs_wer_scatter.png", dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize full evaluation results with WER")
    project_root = Path(__file__).resolve().parents[2]
    csv_in = project_root / "experiments" / "results" / "results.csv"
    out_json = project_root / "experiments" / "results" / "metrics_summary.json"
    plots_dir = project_root / "experiments" / "plots"

    args = parser.parse_args()

    rows = load_rows(csv_in)

    # Global summaries
    rr_wavlm = group_values(rows, "RR", "wavlm_sim")
    cr_wavlm = group_values(rows, "CR", "wavlm_sim")
    ci_wavlm = group_values(rows, "CI", "wavlm_sim")
    rr_wer = group_values(rows, "RR", "wer")
    cr_wer = group_values(rows, "CR", "wer")

    auc, eer = compute_auc_eer(cr_wavlm, ci_wavlm)

    summary: Dict[str, Any] = {
        "global": {
            "wavlm": {
                "RR": summarize(rr_wavlm),
                "CR": summarize(cr_wavlm),
                "CI": summarize(ci_wavlm),
                "CR_RR_ratio_mean": float(np.nan) if not rr_wavlm or not cr_wavlm else (np.nanmean(cr_wavlm) / max(1e-9, np.nanmean(rr_wavlm))),
                "AUC": auc,
                "EER": eer,
            },
            "wer": {
                "RR": summarize(rr_wer),
                "CR": summarize(cr_wer),
            },
        },
        "by_speaker": {},
    }

    # Per-speaker + prompt style summaries
    by_spk: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_spk[r.get("speaker_id", "")].append(r)

    for spk, spk_rows in by_spk.items():
        rr_w = group_values(spk_rows, "RR", "wavlm_sim")
        cr_w = group_values(spk_rows, "CR", "wavlm_sim")
        ci_w = group_values(spk_rows, "CI", "wavlm_sim")
        rr_werr = group_values(spk_rows, "RR", "wer")
        cr_werr = group_values(spk_rows, "CR", "wer")
        auc_s, eer_s = compute_auc_eer(cr_w, ci_w)
        summary["by_speaker"][spk] = {
            "wavlm": {
                "RR": summarize(rr_w),
                "CR": summarize(cr_w),
                "CI": summarize(ci_w),
                "CR_RR_ratio_mean": float(np.nan) if not rr_w or not cr_w else (np.nanmean(cr_w) / max(1e-9, np.nanmean(rr_w))),
                "AUC": auc_s,
                "EER": eer_s,
            },
            "wer": {
                "RR": summarize(rr_werr),
                "CR": summarize(cr_werr),
            },
        }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_boxplots(rows, plots_dir)
    plot_roc(cr_wavlm, ci_wavlm, plots_dir)
    plot_scatter_sim_wer(rows, plots_dir)

    print(f"Wrote metrics summary -> {out_json}")
    print(f"Saved plots -> {plots_dir}")


if __name__ == "__main__":
    main()


