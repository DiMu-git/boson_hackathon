# Automated Prompt Experimentation

This directory contains all scripts, data, and results for the automated prompt experimentation with Higgs Audio v2.

## Directory Structure

```
automated_prompt_experiment/
├── README.md                    # This file
├── experiment_config.json       # Experiment configuration
├── run_experiments.py          # Main experiment runner
├── batch_prompt_eval.py        # Batch prompt evaluation
├── higgs_eval.py               # Higgs model evaluation utilities
├── generate_prompt_texts.py    # Prompt text generation
├── prompts/                    # Input prompt files
├── outputs/                    # Generated audio outputs
├── results/                    # Experiment results
│   ├── higgs_prompt_eval.csv   # Main results CSV
│   └── prompt_sweep/           # Prompt sweep analysis
│       └── results.csv
├── cache/                      # Cached data
│   └── emb_cache/             # Speaker embedding cache
└── instruction/               # Documentation
    ├── cursor_task_en.md
    └── experiment_plan_zh.md
```

## Usage

### Running Experiments

```bash
# Run main experiment
python run_experiments.py

# Run batch evaluation
python batch_prompt_eval.py

# With custom parameters
python run_experiments.py --config experiment_config.json --out-dir outputs --csv results/higgs_prompt_eval.csv
```

### Configuration

Edit `experiment_config.json` to modify:
- System prompts (S1, S2)
- Scene descriptions (D1, D2, D3)
- Decoding parameters (K1-K4)
- Reference audio files (R1, R3)

## Results

- **Main Results**: `results/higgs_prompt_eval.csv`
- **Prompt Sweep**: `results/prompt_sweep/results.csv`
- **Generated Audio**: `outputs/`
- **Cache**: `cache/emb_cache/` (speaker embeddings)
