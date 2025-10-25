## Voice Cloning Evaluation – Brief Report

### Objective
Assess identity fidelity, intelligibility, and safety of cloned voices using limited references, fixed sentences, and two prompt styles.

### Data & Setup
- Dataset: LibriSpeech (top speakers by clip count)
- Per speaker:
  - References: 2–3 clips (6–12 s each, normalized to 16 kHz mono, trimmed, loudness normalized)
  - Reals: 10 normalized clips (non‑overlapping with refs)
  - Impostors: 10 normalized clips from other speakers
- Clones: 10 per speaker for fixed 10 sentences
  - P1 (Neutral): 5 clones
  - P2 (Identity‑Preserving): 5 clones
- System prompt included; [SPEAKER1] tag used for all clones

### Metrics (per pair)
- WavLM cosine (primary identity similarity)
- MFCC20 cosine (timbre spectrum)
- Pitch similarity (median f0)
- Acoustic metric (VoiceAnalyzer overall: pitch + spectral + MFCC)
- WER (ASR) – currently ASR vs ASR (see caveat)

### Pairing
- RR (real vs real, same speaker): 20 pairs
- CR (clone vs real, same speaker): 3 pairs per clone (30 total)
- CI (clone vs impostor speaker): 10 pairs per clone (100 total)

### Key Findings
- Identity fidelity is meaningful:
  - RR (upper bound) WavLM mean ≈ 0.90–0.93
  - CR means are slightly lower (≈ 0.87–0.91) and consistently > CI means (≈ 0.83–0.88)
  - Several settings yield strong separability (AUC up to ~0.96 with low EER)
- Prompt/style patterns repeatedly performing well across speakers:
  - P1 (Neutral) + sentence 5
  - P1 (Neutral) + sentence 3
  - P2 (Identity‑Preserving) + sentence 7

### Best Settings (evidence)
See the generated leaderboards:
- Per‑prompt overall: `experiments/results/prompt_leaderboard.csv`
- Per‑speaker Top‑10: `experiments/results/prompt_leaderboard_top10_per_speaker.csv`
- Recommended settings (frequency‑ranked): `experiments/results/recommended_settings.json`

Example top entries (CR/RR ratio, then AUC):
- 730 · P1+5 · ratio ~0.997 · AUC ~0.96
- 730 · P1+3 · ratio ~0.986 · AUC ~0.63–0.96 range per speaker
- 211 · P2+7 · ratio ~0.977 · AUC ~0.88

### WER Caveat
- Current WER is ASR vs ASR and trends high; for LibriSpeech we should use:
  - RR: reference = official transcript, hypothesis = ASR(real)
  - CR: reference = fixed sentence text, hypothesis = ASR(clone)
This will produce interpretable intelligibility scores.

### Artifacts
- Raw pairs: `experiments/results/results.csv`
- Summaries: `experiments/results/metrics_summary.json`
- Plots: `experiments/plots/` (boxplots, ROC, sim‑vs‑WER)

### Recommendations
- Use P1+5 or P1+3 as default hacking prompts; consider P2+7 when identity preservation is critical.
- Enable text‑based WER for intelligibility analysis.
- Optional: include ECAPA cosine in full protocol CSV (currently available in split‑eval path) for cross‑validator comparison.


