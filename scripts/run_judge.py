"""
EconWritingBench — LLM Judge Runner
Score model outputs using the rubric and LLM judge.
Usage: python scripts/run_judge.py --run_dir results/runs/<run_id>/ --judge_model gpt-4o
"""

import argparse
import json
from pathlib import Path
from openai import OpenAI


def load_rubric() -> dict:
    with open("evaluation/rubrics.json") as f:
        return json.load(f)


def load_judge_prompt() -> str:
    with open("evaluation/judge_prompt.md") as f:
        return f.read()


def judge_sample(client, judge_model: str, system_prompt: str, task: str, sample_output: dict, rubric: dict) -> dict:
    task_rubric = rubric["tasks"][task]
    user_message = f"""
Task type: {task}
Sample ID: {sample_output['id']}
Model: {sample_output['model']}

Rubric dimensions:
{json.dumps(task_rubric, ensure_ascii=False, indent=2)}

Model response to evaluate:
{sample_output['output']}
"""
    response = client.chat.completions.create(
        model=judge_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--judge_model", default="gpt-4o")
    args = parser.parse_args()

    client = OpenAI()
    rubric = load_rubric()
    judge_prompt = load_judge_prompt()
    run_dir = Path(args.run_dir)
    scores_dir = run_dir / "scores"
    scores_dir.mkdir(exist_ok=True)

    for task_file in run_dir.glob("*.json"):
        task = task_file.stem
        if task not in rubric["tasks"]:
            continue
        with open(task_file) as f:
            outputs = json.load(f)
        scored = []
        for sample_output in outputs:
            print(f"Judging {task}/{sample_output['id']}...")
            score = judge_sample(client, args.judge_model, judge_prompt, task, sample_output, rubric)
            scored.append(score)
        out_file = scores_dir / f"{task}_scores.json"
        with open(out_file, "w", ensure_ascii=False) as f:
            json.dump(scored, f, ensure_ascii=False, indent=2)
        print(f"Scores saved to {out_file}")


if __name__ == "__main__":
    main()
