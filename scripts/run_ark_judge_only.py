"""Judge-only runner for EconWritingBench using Volcengine Ark.

Reads an existing inference JSONL (results/runs/*.jsonl) and writes scored JSONL.
Designed to be robust: per-item retry, incremental writes, and resumable.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx
from openai import OpenAI


_JSON_OBJ_RE = re.compile(r"\{[\s\S]*\}")


def extract_json_obj(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = _JSON_OBJ_RE.search(text)
    if not m:
        raise ValueError(f"No JSON object found in response: {text[:1000]}")
    return json.loads(m.group(0))


def ark_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        http_client=httpx.Client(timeout=180.0, trust_env=False),
        max_retries=2,
    )


def ark_chat(client: OpenAI, endpoint_id: str, messages: list[dict[str, str]], max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=endpoint_id,
        messages=messages,
        temperature=0,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


def judge_one(
    *,
    client: OpenAI,
    judge_endpoint: str,
    judge_prompt_text: str,
    rubric: dict[str, Any],
    task: str,
    instruction: str,
    input_obj: dict[str, Any],
    model_output: str,
) -> dict[str, Any]:
    task_spec = rubric["tasks"][task]
    dimensions = task_spec["dimensions"]  # list[{name, weight, description, anchors}]
    hard_deductions = task_spec.get("hard_deductions", [])

    judge_payload = {
        "task": task,
        "scale": rubric.get("scoring_scale", "1-5"),
        "dimensions": dimensions,
        "hard_deductions": hard_deductions,
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
        + json.dumps({"instruction": instruction, "input": input_obj}, ensure_ascii=False)
        + "\n\n"
        + "【模型输出】\n"
        + (model_output or "")
        + "\n\n"
        + "必须输出 JSON，字段包括：dimension_scores、hard_penalties、final_score、rationale。\n"
        + "其中：\n"
        + "- dimension_scores：必须覆盖 rubric.dimensions 里每个维度的 name，取值 1-5 的数字；\n"
        + "- hard_penalties：数组；元素形如 {condition, deduction, evidence}，deduction 为负数或 0；未触发就返回 []；\n"
        + "- final_score：必须是数字（允许小数），范围 1-5；\n"
        + "- rationale：<=200字，必须引用输出中的具体措辞/句子。\n"
        + "严禁输出除 JSON 之外的任何内容。"
    )

    raw = ark_chat(
        client,
        judge_endpoint,
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=800,
    )

    judged = extract_json_obj(raw)

    dim_scores = {d["name"]: float(judged["dimension_scores"][d["name"]]) for d in dimensions}
    weighted = sum(dim_scores[d["name"]] * float(d["weight"]) for d in dimensions)

    # Note: version 1.0 uses deductions (negative values)
    total_deduction = 0.0
    applied_penalties = []
    for p in judged.get("hard_penalties", []) or []:
        total_deduction += float(p.get("deduction", 0))
        applied_penalties.append(p)

    final_score = max(1.0, min(5.0, weighted + total_deduction))

    return {
        "rubric_version": rubric.get("version", "1.0"),
        "dimension_scores": dim_scores,
        "weighted_score": weighted,
        "hard_penalties": applied_penalties,
        "final_score": final_score,
        "rationale": str(judged.get("rationale", ""))[:1200],
        "raw_judge": raw,
    }


def read_done_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                done.add(f"{rec.get('task')}::{rec.get('sample_id')}")
            except Exception:
                continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--api-key", required=True)
    ap.add_argument("--judge-endpoint", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--rubric", required=True)
    ap.add_argument("--judge-prompt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    client = ark_client(args.base_url, args.api_key)

    rubric = json.loads((repo_root / args.rubric).read_text(encoding="utf-8"))
    judge_prompt_text = (repo_root / args.judge_prompt).read_text(encoding="utf-8")

    inp = repo_root / args.run
    outp = repo_root / args.out
    outp.parent.mkdir(parents=True, exist_ok=True)

    done = read_done_ids(outp)

    n_total = 0
    n_done = 0

    with inp.open("r", encoding="utf-8") as fin, outp.open("a", encoding="utf-8") as fout:
        for line in fin:
            rec = json.loads(line)
            key = f"{rec['task']}::{rec['sample_id']}"
            n_total += 1

            if args.limit and n_total > args.limit:
                break

            if key in done:
                n_done += 1
                continue

            last_err = ""
            for attempt in range(1, 4):
                try:
                    judge = judge_one(
                        client=client,
                        judge_endpoint=args.judge_endpoint,
                        judge_prompt_text=judge_prompt_text,
                        rubric=rubric,
                        task=rec["task"],
                        instruction=rec.get("instruction", ""),
                        input_obj=rec.get("input", {}),
                        model_output=rec.get("output", ""),
                    )
                    scored = {**rec, "judge": judge, "judge_model": args.judge_endpoint}
                    fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
                    fout.flush()
                    done.add(key)
                    n_done += 1
                    break
                except Exception as e:
                    last_err = f"{type(e).__name__}: {e}"
                    time.sleep(min(5 * attempt, 15))

            if key not in done:
                # still write a record so aggregation can proceed
                min_s = 1.0
                dims = [d["name"] for d in rubric["tasks"][rec["task"]]["dimensions"]]
                scored = {
                    **rec,
                    "judge": {
                        "rubric_version": rubric.get("version", "1.0"),
                        "dimension_scores": {k: min_s for k in dims},
                        "weighted_score": min_s,
                        "hard_penalties": [],
                        "final_score": min_s,
                        "rationale": f"judge_failed: {last_err}"[:1200],
                        "raw_judge": "",
                    },
                    "judge_model": args.judge_endpoint,
                }
                fout.write(json.dumps(scored, ensure_ascii=False) + "\n")
                fout.flush()
                done.add(key)
                n_done += 1

            if n_done % 5 == 0:
                print(f"progress: {n_done}/{n_total}")

    print(f"done: {n_done}/{n_total}")


if __name__ == "__main__":
    main()
