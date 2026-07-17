"""
EconWritingBench — Inference Runner
Run model completions for all benchmark tasks and save raw outputs.
Usage: python scripts/run_inference.py --model gpt-4o --task all --output_dir results/runs/
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime

TASK_FILES = {
    "related_work": "data/samples/related_work.json",
    "regression_interpretation": "data/samples/regression_interpretation.json",
    "intro_motivation": "data/samples/intro_motivation.json",
    "abstract_polish": "data/samples/abstract_polish.json",
    "research_design": "data/samples/research_design.json",
}

PROMPT_TEMPLATES = {
    "related_work": "data/prompts/task1_related_work.md",
    "regression_interpretation": "data/prompts/task2_regression_interpretation.md",
    "intro_motivation": "data/prompts/task3_intro_motivation.md",
    "abstract_polish": "data/prompts/task4_abstract_polish.md",
    "research_design": "data/prompts/task5_research_design.md",
}


def load_samples(task: str) -> list:
    with open(TASK_FILES[task]) as f:
        return json.load(f)


def build_prompt(sample: dict, template: str) -> str:
    """Build the full prompt for a sample using the task template."""
    instruction = sample["instruction"]
    input_str = json.dumps(sample["input"], ensure_ascii=False, indent=2)
    return f"{template}\n\n## Instruction\n{instruction}\n\n## Input\n```json\n{input_str}\n```\n\n## Your Response"


def run_model(prompt: str, model: str, **kwargs) -> str:
    """Call the appropriate model API. Extend this function for each model."""
    if model.startswith("gpt"):
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1024,
        )
        return response.choices[0].message.content
    elif model.startswith("claude"):
        import anthropic
        client = anthropic.Anthropic()
        message = client.messages.create(
            model=model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    else:
        raise NotImplementedError(f"Model {model} not yet supported. Add implementation in run_model().")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Model name, e.g. gpt-4o, claude-3-5-sonnet-20241022")
    parser.add_argument("--task", default="all", help="Task name or 'all'")
    parser.add_argument("--output_dir", default="results/runs")
    args = parser.parse_args()

    tasks = list(TASK_FILES.keys()) if args.task == "all" else [args.task]
    run_id = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path(args.output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    for task in tasks:
        samples = load_samples(task)
        results = []
        for sample in samples:
            prompt = build_prompt(sample, f"# Task: {task}")
            print(f"Running {task}/{sample['id']} with {args.model}...")
            output = run_model(prompt, args.model)
            results.append({
                "id": sample["id"],
                "task": task,
                "model": args.model,
                "prompt": prompt,
                "output": output,
            })
        out_file = out_dir / f"{task}.json"
        with open(out_file, "w", ensure_ascii=False) as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved {len(results)} results to {out_file}")

    print(f"\nRun complete. Results saved to {out_dir}")


if __name__ == "__main__":
    main()
