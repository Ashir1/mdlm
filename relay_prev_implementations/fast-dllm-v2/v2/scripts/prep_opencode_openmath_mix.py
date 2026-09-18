"""Prepare an OpenCodeInstruct / OpenMathInstruct-2 mixture for Fast-dLLM v2.

The output matches the repo's conversation JSON layout:

    <out_dir>/train_conversation/train_*.json

Paper recipe (Table 2 of *Learned Relay Representations*):

    python scripts/prep_opencode_openmath_mix.py \
      --out_dir data/opencode_openmath_60k_c40m60 \
      --code_rows 24000 --math_rows 36000 --require_code_def

That is the **c40m60** mixture: 24 000 code rows + 36 000 math rows = 60 000
total, 40% code / 60% math. The ``--require_code_def`` flag drops code rows
without a Python ``def`` -- empirically the cleanest filter that still
yields the reported HumanEval / MBPP gains.

Defaults below intentionally do NOT match the paper -- they preserve the
upstream Fast-dLLM v2 example so existing scripts keep working. Pass the
flags above to reproduce Table 2.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from datasets import load_dataset
from transformers import AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lmflow.utils.conversation_template import PRESET_TEMPLATES  # noqa: E402

DEFAULT_LEAK_PATTERNS = (
    "humaneval",
    "human_eval",
    "mbpp",
    "evalplus",
    "math-500",
    "math500",
)


def _parse_csv(arg: str | None) -> list[str]:
    if not arg:
        return []
    return [x.strip() for x in arg.split(",") if x.strip()]


def _parse_boolish_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    return parsed if isinstance(parsed, list) else [parsed]


def _normalize_prompt(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _dedup_key(text: str) -> str:
    return hashlib.sha1(_normalize_prompt(text).encode("utf-8")).hexdigest()


def _has_leak_marker(text: str, patterns: Iterable[str]) -> bool:
    text_l = text.lower()
    return any(p.lower() in text_l for p in patterns)


def _token_len(
    *,
    messages: list[dict[str, str]],
    tokenizer,
    chat_template: str,
) -> int:
    encoded = tokenizer.apply_chat_template(
        conversation=messages,
        chat_template=chat_template,
        return_dict=True,
    )
    return len(encoded["input_ids"])


def _write_shards(out_dir: Path, instances: list[dict[str, Any]], shard_size: int) -> None:
    train_dir = out_dir / "train_conversation"
    train_dir.mkdir(parents=True, exist_ok=True)

    for stale in sorted(train_dir.glob("train_*.json")):
        stale.unlink()
        print(f"Removed stale shard: {stale}")

    n_total = len(instances)
    if shard_size > 0 and shard_size < n_total:
        n_shards = (n_total + shard_size - 1) // shard_size
        for i in range(n_shards):
            chunk = instances[i * shard_size : (i + 1) * shard_size]
            path = train_dir / f"train_shard{i:03d}of{n_shards:03d}_{len(chunk)}.json"
            with path.open("w", encoding="utf-8") as f:
                json.dump({"type": "conversation", "instances": chunk}, f, ensure_ascii=False)
            print(f"Wrote shard {i + 1}/{n_shards}: {len(chunk)} rows -> {path}")
    else:
        path = train_dir / f"train_{n_total}.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump({"type": "conversation", "instances": instances}, f, ensure_ascii=False)
        print(f"Wrote {n_total} rows -> {path}")


def _code_instance(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    prompt = row.get("input")
    answer = row.get("output")
    if prompt is None or answer is None:
        return None

    prompt_s = str(prompt).strip()
    answer_s = str(answer).strip()
    if not prompt_s or not answer_s:
        return None

    if args.require_code_def and "def " not in answer_s:
        return None

    tests = _parse_boolish_list(row.get("unit_tests"))
    if args.require_unit_tests and not tests:
        return None

    try:
        score = float(row.get("average_test_score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    if score < args.min_code_test_score:
        return None

    statuses = [str(x).lower() for x in _parse_boolish_list(row.get("tests_execution_status"))]
    if args.require_all_code_tests_pass and statuses and any(s != "pass" for s in statuses):
        return None

    if _has_leak_marker(f"{prompt_s}\n{answer_s}", args.leak_patterns):
        return None

    return {
        "messages": [
            {"role": "user", "content": prompt_s},
            {"role": "assistant", "content": answer_s},
        ],
        "source": "opencodeinstruct",
    }


def _math_instance(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any] | None:
    problem = row.get("problem")
    solution = row.get("generated_solution") or row.get("solution")
    if problem is None or solution is None:
        return None

    problem_s = str(problem).strip()
    solution_s = str(solution).strip()
    if not problem_s or not solution_s:
        return None
    if len(problem_s) > args.max_math_problem_chars:
        return None
    if len(solution_s) > args.max_math_solution_chars:
        return None
    if _has_leak_marker(f"{problem_s}\n{solution_s}", args.leak_patterns):
        return None

    return {
        "messages": [
            {"role": "user", "content": problem_s},
            {"role": "assistant", "content": solution_s},
        ],
        "source": "openmathinstruct2",
    }


def _collect(
    *,
    label: str,
    dataset_id: str,
    split: str,
    target: int,
    seed: int,
    shuffle_buffer: int,
    tokenizer,
    chat_template: str,
    args: argparse.Namespace,
    row_to_instance,
    seen_prompts: set[str],
) -> list[dict[str, Any]]:
    print(f"Loading {label}: {dataset_id} split={split!r} streaming=True")
    ds = load_dataset(dataset_id, split=split, streaming=True)
    if shuffle_buffer > 0:
        ds = ds.shuffle(seed=seed, buffer_size=shuffle_buffer)

    out: list[dict[str, Any]] = []
    scanned = 0
    skipped_length = 0
    skipped_dup = 0
    skipped_filter = 0
    for row in ds:
        scanned += 1
        inst = row_to_instance(row, args)
        if inst is None:
            skipped_filter += 1
            continue

        prompt = inst["messages"][0]["content"]
        key = _dedup_key(prompt)
        if key in seen_prompts:
            skipped_dup += 1
            continue

        try:
            length = _token_len(
                messages=inst["messages"],
                tokenizer=tokenizer,
                chat_template=chat_template,
            )
        except Exception as exc:  # noqa: BLE001 - keep streaming robust and log the cause.
            skipped_filter += 1
            if skipped_filter <= 5:
                print(f"[{label}] skipped tokenization failure: {exc}")
            continue

        if length > args.block_size:
            skipped_length += 1
            continue

        inst["tokenized_chat_len"] = length
        seen_prompts.add(key)
        out.append(inst)
        if len(out) % args.log_every == 0:
            print(
                f"[{label}] accepted={len(out)}/{target} scanned={scanned} "
                f"skip_filter={skipped_filter} skip_length={skipped_length} skip_dup={skipped_dup}"
            )
        if len(out) >= target:
            break

    if len(out) < target:
        raise RuntimeError(
            f"{label}: collected {len(out)} rows but target is {target}. "
            f"Scanned {scanned}; skipped filter={skipped_filter}, length={skipped_length}, dup={skipped_dup}."
        )

    print(
        f"[{label}] done: accepted={len(out)} scanned={scanned} "
        f"skip_filter={skipped_filter} skip_length={skipped_length} skip_dup={skipped_dup}"
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out_dir", default="data/opencode_openmath_60k")
    p.add_argument("--code_dataset", default="nvidia/OpenCodeInstruct")
    p.add_argument("--code_split", default="train")
    p.add_argument("--math_dataset", default="nvidia/OpenMathInstruct-2")
    p.add_argument("--math_split", default="train_1M")
    p.add_argument("--code_rows", type=int, default=33000)
    p.add_argument("--math_rows", type=int, default=27000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--code_shuffle_buffer", type=int, default=100000)
    p.add_argument("--math_shuffle_buffer", type=int, default=50000)
    p.add_argument("--block_size", type=int, default=2048)
    p.add_argument("--model_name_or_path", default="Efficient-Large-Model/Fast_dLLM_v2_1.5B")
    p.add_argument("--conversation_template", default="fast_dllm_v2")
    p.add_argument("--shard_size", type=int, default=50000)
    p.add_argument("--min_code_test_score", type=float, default=1.0)
    p.add_argument("--require_unit_tests", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require_all_code_tests_pass", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--require_code_def", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--max_math_problem_chars", type=int, default=4000)
    p.add_argument("--max_math_solution_chars", type=int, default=6000)
    p.add_argument("--leak_patterns", default=",".join(DEFAULT_LEAK_PATTERNS))
    p.add_argument("--log_every", type=int, default=1000)
    args = p.parse_args()

    args.leak_patterns = _parse_csv(args.leak_patterns)

    if args.conversation_template not in PRESET_TEMPLATES:
        raise SystemExit(f"Unknown conversation template: {args.conversation_template}")
    chat_template = PRESET_TEMPLATES[args.conversation_template]
    if not isinstance(chat_template, str):
        raise SystemExit(
            f"{args.conversation_template} is not a Jinja chat template; "
            "this script expects tokenizer.apply_chat_template compatibility."
        )

    print(f"Loading tokenizer: {args.model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=False)

    seen_prompts: set[str] = set()
    code = _collect(
        label="code",
        dataset_id=args.code_dataset,
        split=args.code_split,
        target=args.code_rows,
        seed=args.seed,
        shuffle_buffer=args.code_shuffle_buffer,
        tokenizer=tokenizer,
        chat_template=chat_template,
        args=args,
        row_to_instance=_code_instance,
        seen_prompts=seen_prompts,
    )
    math = _collect(
        label="math",
        dataset_id=args.math_dataset,
        split=args.math_split,
        target=args.math_rows,
        seed=args.seed + 1,
        shuffle_buffer=args.math_shuffle_buffer,
        tokenizer=tokenizer,
        chat_template=chat_template,
        args=args,
        row_to_instance=_math_instance,
        seen_prompts=seen_prompts,
    )

    rng = random.Random(args.seed)
    instances = code + math
    rng.shuffle(instances)

    # Reproducibility check: warn loudly if the resulting mixture is not the
    # 60k c40m60 split used to compute Table 2 of the paper. The targets are
    # only enforced when the user passes the c40m60 defaults
    # (--code_rows 24000 --math_rows 36000); other ratios are allowed but
    # will not match the reported numbers.
    if args.code_rows == 24000 and args.math_rows == 36000:
        if len(code) < args.code_rows:
            print(
                f"[warn] paper c40m60 mixture asked for {args.code_rows} code rows "
                f"but only {len(code)} survived filtering -- Table 2 numbers may "
                "not reproduce. Re-run with a larger --code_shuffle_buffer or "
                "loosen --require_unit_tests / --min_code_test_score / "
                "--require_code_def to recover.",
                file=sys.stderr,
            )
        if len(math) < args.math_rows:
            print(
                f"[warn] paper c40m60 mixture asked for {args.math_rows} math rows "
                f"but only {len(math)} survived filtering -- Table 2 numbers may "
                "not reproduce. Re-run with a larger --math_shuffle_buffer or "
                "loosen --max_math_problem_chars / --max_math_solution_chars.",
                file=sys.stderr,
            )

    out_dir = Path(args.out_dir)
    _write_shards(out_dir, instances, args.shard_size)

    summary = {
        "code_dataset": args.code_dataset,
        "code_split": args.code_split,
        "code_rows": len(code),
        "math_dataset": args.math_dataset,
        "math_split": args.math_split,
        "math_rows": len(math),
        "total_rows": len(instances),
        "block_size": args.block_size,
        "conversation_template": args.conversation_template,
        "model_name_or_path": args.model_name_or_path,
        "seed": args.seed,
    }
    summary_path = out_dir / "prep_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f"Wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
