# Contributing to EconWritingBench

We welcome contributions of all kinds!

## How to Contribute

### 1. Add New Evaluation Samples
- Place new samples in `data/samples/<task>.json` following the existing schema
- Each sample must include: `id`, `task`, `instruction`, `input`, `reference_output`, `evaluation_focus`
- Open a PR with a brief description of why the new sample tests a distinct aspect

### 2. Add New Model Evaluations
- Run the full pipeline (`run_inference.py` + `run_judge.py` + `aggregate_scores.py`)
- Add your results to `results/leaderboard.csv`
- Include model version, API temperature, and judge model in `results/runs/<your_run>/README.md`

### 3. Improve Rubrics
- Open an issue describing the proposed change and rationale
- Rubric changes require discussion to maintain comparability across runs

## Code of Conduct

Be respectful and constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
