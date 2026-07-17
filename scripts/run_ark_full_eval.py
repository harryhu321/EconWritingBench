"""Run EconWritingBench full eval on Volcengine Ark endpoints.

Steps:
1) Validate inference endpoint is reachable.
2) Run inference for all samples (75).
3) Use a separate judge endpoint to score each output with rubric.
4) Aggregate task-level + weighted overall scores.
5) Write leaderboard CSV + a single Markdown row snippet.

Notes:
- Uses OpenAI Python SDK with base_url pointed to Ark.
- Runs inference/judging concurrently (default 5 workers) to reduce wall time.
- Writes JSONL incrementally so the run is resumable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI


TASK_FILES = [
    "related_work.json",
    "regression_interpretation.json",
    "intro_motivation.json",
    "abstract_polish.json",
    "research_design.json",
]

# User-specified task weights
OVERALL_TASK_WEIGHTS = {
    "related_work": 0.20,
    "regression_interpretation": 0.25,
    "intro_motivation": 0.20,
    "abstract_polish": 0.15,
    "research_design": 0.20,
}


@dataclass(frozen=True)
class Sample:
    task: str
    sample_id: str
    instruction: str
    input_obj: dict[str, Any]


def load_samples(samples_dir: Path) -> list[Sample]:
    """Load samples from data/samples.

    This repo stores each task file as a JSON array, where each element is a sample
    object containing: id, task, instruction, input.
    """

    samples: list[Sample] = []
    for name in TASK_FILES:
        fp = samples_dir / name
        data = json.loads(fp.read_text(encoding="utf-8"))

        if isinstance(data, list):
            for s in data:
                samples.append(
                    Sample(
                        task=str(s["task"]),
                        sample_id=str(s.get("sample_id") or s.get("id")),
                        instruction=str(s["instruction"]),
                        input_obj=dict(s["input"]),
                    )
                )
        else:
            # Backward-compatible format: {"task": ..., "samples": [...]}
            task = str(data["task"])
            for s in data["samples"]:
                samples.append(
                    Sample(
                        task=task,
                        sample_id=str(s.get("sample_id") or s.get("id")),
                        instruction=str(s["instruction"]),
                        input_obj=dict(s["input"]),
                    )
                )

    return samples


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def extract_json_obj(text: str) -> dict[str, Any]:
    """Try strict json.loads first; fallback to extracting the first {...} block."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass

    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in response: {text[:2000]}")
    return json.loads(m.group(0))


_thread_local = threading.local()


def get_client(*, base_url: str, api_key: str) -> OpenAI:
    """Get a per-thread OpenAI client.

    We set trust_env=False to avoid inheriting proxy-related env vars (NO_PROXY may
    include '::1', which can break httpx URL parsing).

    Also set a slightly longer timeout and limited retries for flaky endpoints.
    """

    if getattr(_thread_local, "client", None) is None:
        http_client = httpx.Client(timeout=180.0, trust_env=False)
        _thread_local.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=http_client,
            max_retries=2,
        )
    return _thread_local.client


def ark_chat(
    *,
    base_url: str,
    api_key: str,
    endpoint_id: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
) -> str:
    client = get_client(base_url=base_url, api_key=api_key)
    resp = client.chat.completions.create(
        model=endpoint_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def format_prompt(sample: Sample) -> str:
    return (
        sample.instruction
        + "\n\n"
        + "【输入】\n"
        + json.dumps(sample.input_obj, ensure_ascii=False, indent=2)
    )


def validate_endpoint(*, base_url: str, api_key: str, endpoint_id: str) -> str:
    return ark_chat(
        base_url=base_url,
        api_key=api_key,
        endpoint_id=endpoint_id,
        messages=[
            {"role": "system", "content": "你是一个乐于助人的助手。"},
            {"role": "user", "content": "你好！请用一句话回复：endpoint OK"},
        ],
        temperature=0,
        max_tokens=32,
    )


def judge_one(
    *,
    base_url: str,
    api_key: str,
    judge_endpoint: str,
    judge_prompt_text: str,
    rubric: dict[str, Any],
    sample: Sample,
    model_output: str,
) -> dict[str, Any]:
    task_spec = rubric["tasks"][sample.task]
    weights: dict[str, float] = task_spec["weights"]
    dimensions: dict[str, str] = task_spec["dimensions"]
    hard_penalties: list[dict[str, Any]] = task_spec.get("hard_penalties", [])

    judge_payload = {
        "task": sample.task,
        "task_name": task_spec["name"],
        "scale": rubric["scale"],
        "dimension_weights": weights,
        "dimensions": dimensions,
        "hard_penalties": hard_penalties,
    }

    system = (
        "你是一个严格的经济学论文写作评审。"
        "你必须只输出可被 json.loads 解析的 JSON（不要 Markdown，不要多余文本）。"
        "理由必须引用输出中的具体句子/措辞，避免空泛。"
    )

    user = (
        judge_prompt_text.strip()
        + "\n\n"
        + "【Rubric】\n"
        + json.dumps(judge_payload, ensure_ascii=False)
        + "\n\n"
        + "【任务输入】\n"
        + json.dumps({"instruction": sample.instruction, "input": sample.input_obj}, ensure_ascii=False)
        + "\n\n"
        + "【模型输出】\n"
        + (model_output or "")
        + "\n\n"
        + "必须输出 JSON，字段包括：dimension_scores、hard_penalties、final_score、rationale。\n"
        + "其中：\n"
        + "- dimension_scores：必须覆盖 rubric.dimensions 的所有 key，取值 1-5 的数字；\n"
        + "- hard_penalties：数组；name 只能从 rubric.hard_penalties 的 name 中选；未触发就返回 []；\n"
        + "- final_score：必须是数字（允许小数），表示最终得分（1-5）；\n"
        + "- rationale：<=200字，必须引用输出中的具体措辞/句子。\n"
        + "严禁输出除 JSON 之外的任何内容。"
    )

    raw = ark_chat(
        base_url=base_url,
        api_key=api_key,
        endpoint_id=judge_endpoint,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_tokens=700,
    )

    judged = extract_json_obj(raw)

    # Validate + compute weighted score locally (trust rubric weights)
    dim_scores: dict[str, float] = {k: float(judged["dimension_scores"][k]) for k in weights.keys()}
    weighted = sum(dim_scores[k] * float(weights[k]) for k in weights.keys())

    penalty_caps = {p["name"]: float(p["max_deduction"]) for p in hard_penalties}
    total_deduction = 0.0
    applied_penalties: list[dict[str, Any]] = []
    for p in judged.get("hard_penalties", []) or []:
        name = p.get("name")
        if name not in penalty_caps:
            continue
        deduction = float(p.get("deduction", 0))
        deduction = max(0.0, min(deduction, penalty_caps[name]))
        if deduction <= 0:
            continue
        total_deduction += deduction
        applied_penalties.append(
            {
                "name": name,
                "deduction": deduction,
                "evidence": str(p.get("evidence", ""))[:500],
            }
        )

    min_s = float(rubric["scale"]["min"])
    max_s = float(rubric["scale"]["max"])

    try:
        judge_final = float(judged.get("final_score"))
    except Exception:
        judge_final = weighted - total_deduction

    final_score = max(min_s, min(max_s, judge_final))

    return {
        "rubric_version": rubric["rubric_version"],
        "dimension_scores": dim_scores,
        "weighted_score": weighted,
        "hard_penalties": applied_penalties,
        "total_deduction": total_deduction,
        "final_score": final_score,
        "rationale": str(judged.get("rationale", ""))[:1200],
        "raw_judge": raw,
    }


def jsonl_read_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(f"{rec.get('task')}::{rec.get('sample_id')}")
            except Exception:
                continue
    return done


def write_progress(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--model-endpoint", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--judge-endpoint", required=True)
    ap.add_argument("--samples", default="data/samples")
    ap.add_argument("--rubric", default="evaluation/rubrics.json")
    ap.add_argument("--judge-prompt", default="evaluation/judge_prompt.md")
    ap.add_argument("--out-run", required=True)
    ap.add_argument("--out-scored", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]

    rubric = json.loads((repo_root / args.rubric).read_text(encoding="utf-8"))
    judge_prompt_text = (repo_root / args.judge_prompt).read_text(encoding="utf-8")
    samples = load_samples(repo_root / args.samples)

    # Step 1: validate endpoint
    test_resp = validate_endpoint(base_url=args.base_url, api_key=args.api_key, endpoint_id=args.model_endpoint)
    print(f"[validate] {args.model_endpoint}: {test_resp.strip()[:200]}")
    if args.validate_only:
        return

    out_run = repo_root / args.out_run
    out_scored = repo_root / args.out_scored
    out_csv = repo_root / args.out_csv
    out_md = repo_root / args.out_md

    progress_path = out_run.with_suffix(".progress.json")

    done_infer = jsonl_read_existing_ids(out_run)
    done_scored = jsonl_read_existing_ids(out_scored)

    # Step 2: inference (concurrent)
    out_run.parent.mkdir(parents=True, exist_ok=True)

    def infer_worker(s: Sample) -> dict[str, Any]:
        last_err = ""
        for attempt in range(1, 4):
            try:
                output = ark_chat(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    endpoint_id=args.model_endpoint,
                    messages=[
                        {"role": "system", "content": "你是一个经济学论文写作助手。严格遵循指令输出。"},
                        {"role": "user", "content": format_prompt(s)},
                    ],
                    temperature=0,
                    max_tokens=args.max_tokens,
                )
                return {
                    "task": s.task,
                    "sample_id": s.sample_id,
                    "model": args.model_name,
                    "input": s.input_obj,
                    "instruction": s.instruction,
                    "output": output,
                    "date": "2026-07-17",
                }
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(min(5 * attempt, 15))

        # Mark failure but keep pipeline moving
        return {
            "task": s.task,
            "sample_id": s.sample_id,
            "model": args.model_name,
            "input": s.input_obj,
            "instruction": s.instruction,
            "output": "",
            "error": last_err,
            "date": "2026-07-17",
        }

    pending_infer = [s for s in samples if f"{s.task}::{s.sample_id}" not in done_infer]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 5))) as ex, out_run.open("a", encoding="utf-8") as fout:
        futures = {ex.submit(infer_worker, s): s for s in pending_infer}
        for k, fut in enumerate(as_completed(futures), 1):
            s = futures[fut]
            rec = fut.result()
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            done_infer.add(f"{s.task}::{s.sample_id}")

            if k % 5 == 0 or len(done_infer) == len(samples):
                write_progress(
                    progress_path,
                    {
                        "stage": "inference",
                        "done": len(done_infer),
                        "total": len(samples),
                        "elapsed_sec": round(time.time() - t0, 1),
                    },
                )

    # Build an index from inference file (to support resume)
    infer_records: dict[str, dict[str, Any]] = {}
    with out_run.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            infer_records[f"{rec['task']}::{rec['sample_id']}"] = rec

    # Step 3: judge scoring (concurrent)
    out_scored.parent.mkdir(parents=True, exist_ok=True)

    def judge_worker(s: Sample) -> dict[str, Any]:
        key = f"{s.task}::{s.sample_id}"
        rec = infer_records[key]

        # If inference failed, skip judging (final_score will be min score)
        if rec.get("error"):
            min_s = float(rubric["scale"]["min"])
            judge = {
                "rubric_version": rubric["rubric_version"],
                "dimension_scores": {k: min_s for k in rubric["tasks"][s.task]["weights"].keys()},
                "weighted_score": min_s,
                "hard_penalties": [],
                "total_deduction": 0.0,
                "final_score": min_s,
                "rationale": f"inference_failed: {rec.get('error', '')}"[:1200],
                "raw_judge": "",
            }
            return {**rec, "judge": judge, "judge_model": args.judge_endpoint}

        last_err = ""
        for attempt in range(1, 4):
            try:
                judge = judge_one(
                    base_url=args.base_url,
                    api_key=args.api_key,
                    judge_endpoint=args.judge_endpoint,
                    judge_prompt_text=judge_prompt_text,
                    rubric=rubric,
                    sample=s,
                    model_output=rec.get("output", ""),
                )
                return {**rec, "judge": judge, "judge_model": args.judge_endpoint}
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(min(5 * attempt, 15))

        # Mark judge failure but keep record
        min_s = float(rubric["scale"]["min"])
        judge = {
            "rubric_version": rubric["rubric_version"],
            "dimension_scores": {k: min_s for k in rubric["tasks"][s.task]["weights"].keys()},
            "weighted_score": min_s,
            "hard_penalties": [],
            "total_deduction": 0.0,
            "final_score": min_s,
            "rationale": f"judge_failed: {last_err}"[:1200],
            "raw_judge": "",
        }
        return {**rec, "judge": judge, "judge_model": args.judge_endpoint}

    pending_scored = [s for s in samples if f"{s.task}::{s.sample_id}" not in done_scored and f"{s.task}::{s.sample_id}" in infer_records]
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, min(args.workers, 5))) as ex, out_scored.open("a", encoding="utf-8") as fout:
        futures = {ex.submit(judge_worker, s): s for s in pending_scored}
        for k, fut in enumerate(as_completed(futures), 1):
            s = futures[fut]
            scored = fut.result()
            fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
            fout.flush()
            done_scored.add(f"{s.task}::{s.sample_id}")

            if k % 5 == 0 or len(done_scored) == len(samples):
                write_progress(
                    progress_path,
                    {
                        "stage": "judge",
                        "done": len(done_scored),
                        "total": len(samples),
                        "elapsed_sec": round(time.time() - t1, 1),
                    },
                )

    # Step 4: aggregate
    task_to_scores: dict[str, list[float]] = {t: [] for t in rubric["tasks"].keys()}
    with out_scored.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            task_to_scores[rec["task"]].append(float(rec["judge"]["final_score"]))

    task_avgs = {t: (sum(v) / len(v) if v else 0.0) for t, v in task_to_scores.items()}
    overall = sum(task_avgs[t] * OVERALL_TASK_WEIGHTS[t] for t in OVERALL_TASK_WEIGHTS.keys())

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "overall",
                "related_work",
                "regression_interpretation",
                "intro_motivation",
                "abstract_polish",
                "research_design",
                "rubric_version",
                "judge_endpoint",
                "date",
                "run",
            ],
        )
        w.writeheader()
        w.writerow(
            {
                "model": args.model_name,
                "overall": round(overall, 4),
                **{k: round(v, 4) for k, v in task_avgs.items()},
                "rubric_version": rubric["rubric_version"],
                "judge_endpoint": args.judge_endpoint,
                "date": "2026-07-17",
                "run": Path(args.out_run).name,
            }
        )

    def fmt2(x: float) -> str:
        return f"{x:.2f}"

    row = (
        f"| {args.model_name} | {fmt2(overall)} | {fmt2(task_avgs['related_work'])} | "
        f"{fmt2(task_avgs['regression_interpretation'])} | {fmt2(task_avgs['intro_motivation'])} | "
        f"{fmt2(task_avgs['abstract_polish'])} | {fmt2(task_avgs['research_design'])} | "
        f"rubric-{rubric['rubric_version']} | 2026-07-17 |\n"
    )
    out_md.write_text(row, encoding="utf-8")

    write_progress(
        progress_path,
        {
            "stage": "done",
            "inference_records": len(done_infer),
            "scored_records": len(done_scored),
            "overall": round(overall, 4),
            "task_avgs": {k: round(v, 4) for k, v in task_avgs.items()},
            "artifacts": {
                "run_jsonl": str(out_run),
                "scored_jsonl": str(out_scored),
                "leaderboard_csv": str(out_csv),
                "leaderboard_md": str(out_md),
                "progress": str(progress_path),
            },
        },
    )

    print(json.dumps({"overall": overall, "task_avgs": task_avgs}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
