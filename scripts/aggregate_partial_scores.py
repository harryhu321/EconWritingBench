"""Aggregate a (possibly partial) scored JSONL into leaderboard CSV/MD.

This repo's scored JSONL contains records with:
- task
- sample_id
- model
- judge.final_score

We compute per-task averages on the available scored samples for that task.
Overall is weighted by user-specified task weights.

This script is intended for "best effort" reporting when some judge calls time out.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


TASK_ORDER = [
    ("related_work", "Task1 Related Work"),
    ("regression_interpretation", "Task2 回归解读"),
    ("intro_motivation", "Task3 Introduction"),
    ("abstract_polish", "Task4 摘要润色"),
    ("research_design", "Task5 研究设计"),
]

WEIGHTS = {
    "related_work": 0.20,
    "regression_interpretation": 0.25,
    "intro_motivation": 0.20,
    "abstract_polish": 0.15,
    "research_design": 0.20,
}


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--judge", required=True)
    ap.add_argument("--date", required=True)
    args = ap.parse_args()

    scored_path = Path(args.scored)

    task_scores: dict[str, list[float]] = defaultdict(list)
    task_counts: dict[str, int] = defaultdict(int)

    with scored_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            t = rec["task"]
            fs = rec.get("judge", {}).get("final_score")
            if fs is None:
                continue
            task_scores[t].append(float(fs))
            task_counts[t] += 1

    task_avgs = {t: mean(task_scores.get(t, [])) for t, _ in TASK_ORDER}

    # For partial runs, only aggregate over tasks that have any scored samples.
    active_tasks = [t for t, _ in TASK_ORDER if task_counts.get(t, 0) > 0]
    denom = sum(WEIGHTS[t] for t in active_tasks) if active_tasks else 0.0
    overall = (
        sum(task_avgs[t] * WEIGHTS[t] for t in active_tasks) / denom
        if denom > 0
        else 0.0
    )

    # CSV
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    headers = [
        "模型",
        "Overall",
        *[name for _, name in TASK_ORDER],
        "版本",
        "日期",
        "样本覆盖",
    ]

    total_scored = sum(task_counts.get(t, 0) for t, _ in TASK_ORDER)
    coverage = " / ".join([f"{t}:{task_counts.get(t,0)}" for t, _ in TASK_ORDER])

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerow(
            [
                args.model_name,
                f"{overall:.2f}",
                *[f"{task_avgs[t]:.2f}" for t, _ in TASK_ORDER],
                args.judge,
                args.date,
                f"{total_scored}/75 ({coverage})",
            ]
        )

    # MD row (for README table with same columns as results/leaderboard_2026-07-17.csv)
    row = (
        f"| {args.model_name} | {overall:.2f} | {task_avgs['related_work']:.2f} | "
        f"{task_avgs['regression_interpretation']:.2f} | {task_avgs['intro_motivation']:.2f} | "
        f"{task_avgs['abstract_polish']:.2f} | {task_avgs['research_design']:.2f} | {args.judge} | {args.date} |\n"
    )

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(row, encoding="utf-8")

    print(json.dumps({
        "overall": overall,
        "task_avgs": task_avgs,
        "task_counts": dict(task_counts),
        "total_scored": total_scored,
        "out_csv": str(out_csv),
        "out_md": str(out_md),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
