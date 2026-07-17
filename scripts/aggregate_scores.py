"""
EconWritingBench — Score Aggregator
Aggregate per-sample scores into task scores and overall leaderboard.
Usage: python scripts/aggregate_scores.py --run_dir results/runs/<run_id>/
"""

import argparse
import json
import csv
from pathlib import Path


TASK_WEIGHTS = {
    "related_work": 0.20,
    "regression_interpretation": 0.25,
    "intro_motivation": 0.20,
    "abstract_polish": 0.15,
    "research_design": 0.20,
}


def aggregate_task_scores(scores_dir: Path) -> dict:
    task_avgs = {}
    for score_file in scores_dir.glob("*_scores.json"):
        task = score_file.stem.replace("_scores", "")
        with open(score_file) as f:
            scores = json.load(f)
        task_score = sum(s.get("weighted_score", 0) for s in scores) / len(scores)
        task_avgs[task] = round(task_score, 3)
    return task_avgs


def compute_overall(task_avgs: dict) -> float:
    total_weight = 0
    weighted_sum = 0
    for task, weight in TASK_WEIGHTS.items():
        if task in task_avgs:
            weighted_sum += task_avgs[task] * weight
            total_weight += weight
    return round(weighted_sum / total_weight, 3) if total_weight > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    scores_dir = run_dir / "scores"
    task_avgs = aggregate_task_scores(scores_dir)
    overall = compute_overall(task_avgs)

    model = run_dir.name.rsplit("_", 2)[0]
    row = {"model": model, "overall": overall, **task_avgs}

    print("\n=== Score Summary ===")
    for k, v in row.items():
        print(f"  {k}: {v}")

    leaderboard_path = Path("results/leaderboard.csv")
    write_header = not leaderboard_path.exists()
    with open(leaderboard_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f"\nLeaderboard updated: {leaderboard_path}")


if __name__ == "__main__":
    main()
