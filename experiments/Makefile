### Commands for automated prompt experiment

run_automated_prompt_experiment:
	uv run automated_prompt_experiment/run_experiments.py

run_batch_prompt_eval:
	uv run python automated_prompt_experiment/batch_prompt_eval.py \
		--dataset librispeech --speaker-id 211 \
		--min-clips 40 --seed 42 --train-frac 0.05 \
		--sample-rate 24000 --max-train-clones 11 --eval-downsample-frac 0.01 \
		--prompts-glob "automated_prompt_experiment/prompts/text_*.txt"

run_higgs_eval:
	uv run automated_prompt_experiment/higgs_eval.py

run_generate_prompt_texts:
	uv run automated_prompt_experiment/generate_prompt_texts.py

### Decoding grid experiment
run_decoding_grid_small:
	uv run python decoding_grid_experiment/run_decoding_grid.py --grid small

run_decoding_grid_large:
	uv run python decoding_grid_experiment/run_decoding_grid.py --grid large

analyze_decoding_grid:
	uv run python decoding_grid_experiment/analyze_decoding_results.py