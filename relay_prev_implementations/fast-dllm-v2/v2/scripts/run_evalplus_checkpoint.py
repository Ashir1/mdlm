#!/usr/bin/env python3
"""Run EvalPlus for one checkpoint and optionally log metrics to W&B."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def _split_words(s: str) -> list[str]:
    return [x for x in s.replace(",", " ").split() if x]


def _result_path_for_jsonl(jsonl: Path) -> Path:
    dotted = jsonl.with_suffix(jsonl.suffix + ".eval_results.json")
    if dotted.is_file():
        return dotted
    legacy = jsonl.with_name(jsonl.stem + "_eval_results.json")
    if legacy.is_file():
        return legacy
    return dotted


def _pass_rates_from_per_task_eval(ev: Any) -> tuple[float | None, float | None, int]:
    if not isinstance(ev, dict) or not ev:
        return None, None, 0
    n_ok = 0
    n_base = 0
    n_plus = 0
    for _task_id, rows in ev.items():
        if not rows:
            continue
        row = rows[0] if isinstance(rows, (list, tuple)) else rows
        if not isinstance(row, dict):
            continue
        n_ok += 1
        if row.get("base_status") == "pass":
            n_base += 1
        if row.get("plus_status") == "pass":
            n_plus += 1
    if n_ok == 0:
        return None, None, 0
    return n_base / n_ok, n_plus / n_ok, n_ok


def _load_pass_at_1(path: Path) -> dict[str, float | int | None]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    p = data.get("pass_at_k") or {}
    base = p.get("base") or {}
    plus = p.get("plus") or {}
    p1_base = base.get("pass@1")
    p1_plus = plus.get("pass@1") if plus else None
    ev = data.get("eval")
    n_tasks = len(ev) if isinstance(ev, dict) else 0
    if p1_base is None and isinstance(ev, dict) and ev:
        p1_base, p1_plus, n_tasks = _pass_rates_from_per_task_eval(ev)
    return {
        "base_pass_at_1": p1_base,
        "plus_pass_at_1": p1_plus,
        "n_tasks": n_tasks,
    }


def _checkpoint_label_and_step(model_path: Path, fallback_step: int | None) -> tuple[str, int | None]:
    label = model_path.name
    m = re.fullmatch(r"checkpoint-(\d+)", label)
    if m:
        return label, int(m.group(1))
    return label, fallback_step


def _resolve_block_cache(mode: str, use_carry: bool) -> bool:
    mode = mode.lower()
    if mode not in {"auto", "on", "off", "1", "0", "true", "false"}:
        raise ValueError("--block_cache_mode must be one of auto/on/off")
    if mode in {"off", "0", "false"}:
        return False
    if mode in {"on", "1", "true"}:
        if use_carry:
            print(
                "[run_evalplus_checkpoint] carry-aware block cache is not implemented; "
                "forcing block cache off for this carry checkpoint.",
                file=sys.stderr,
            )
            return False
        return True
    return not use_carry


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


def _maybe_log_wandb(args: argparse.Namespace, metrics: dict[str, float | int | None]) -> None:
    if args.no_wandb:
        return
    try:
        import wandb
    except Exception as e:
        print(f"[run_evalplus_checkpoint] wandb unavailable; skip logging: {e}", file=sys.stderr)
        return

    tags = _split_words(args.wandb_tags)
    tags.extend(["evalplus", args.checkpoint_label])
    if args.use_carry:
        tags.append("use_carry")
    if args.use_block_cache_resolved:
        tags.append("block_cache")

    run = wandb.init(
        project=args.wandb_project or os.environ.get("WANDB_PROJECT"),
        entity=args.wandb_entity or os.environ.get("WANDB_ENTITY"),
        group=args.wandb_group or None,
        name=args.wandb_name or f"evalplus__{args.run_name}__{args.checkpoint_label}",
        job_type="evalplus",
        tags=tags,
        config={
            "model_path": args.model_path,
            "checkpoint_label": args.checkpoint_label,
            "checkpoint_step": args.checkpoint_step,
            "datasets": args.datasets,
            "seeds": args.seeds,
            "threshold": args.threshold,
            "use_carry": args.use_carry,
            "block_cache_mode": args.block_cache_mode,
            "use_block_cache_resolved": args.use_block_cache_resolved,
        },
    )
    try:
        step = args.checkpoint_step if args.checkpoint_step is not None else None
        run.log(metrics, step=step)
    finally:
        run.finish()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--checkpoint_step", type=int, default=None)
    ap.add_argument("--datasets", default="humaneval mbpp")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--bd_size", type=int, default=32)
    ap.add_argument("--small_block_size", type=int, default=8)
    ap.add_argument("--use_carry", action="store_true")
    ap.add_argument("--block_cache_mode", default="auto", choices=("auto", "on", "off"))
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--wandb_project", default="")
    ap.add_argument("--wandb_entity", default="")
    ap.add_argument("--wandb_group", default="")
    ap.add_argument("--wandb_name", default="")
    ap.add_argument("--wandb_tags", default="")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    model_path = Path(args.model_path)
    args.checkpoint_label, inferred_step = _checkpoint_label_and_step(model_path, args.checkpoint_step)
    args.checkpoint_step = inferred_step
    args.use_block_cache_resolved = _resolve_block_cache(args.block_cache_mode, args.use_carry)

    datasets = _split_words(args.datasets)
    seeds = _split_words(args.seeds)
    if not datasets:
        raise SystemExit("--datasets produced no dataset names")
    if not seeds:
        raise SystemExit("--seeds produced no seed values")

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, float | int | None] = {
        "checkpoint_step": args.checkpoint_step,
        "use_carry": int(args.use_carry),
        "use_block_cache": int(args.use_block_cache_resolved),
    }

    for dataset in datasets:
        if dataset not in {"humaneval", "mbpp"}:
            raise SystemExit(f"Unsupported EvalPlus dataset: {dataset}")
        for seed_s in seeds:
            seed = int(seed_s)
            stem = f"{args.run_name}__{args.checkpoint_label}__{dataset}__s{seed}"
            jsonl = out_dir / f"{stem}.jsonl"
            result_path = _result_path_for_jsonl(jsonl)

            if args.force or not jsonl.is_file():
                gen_cmd = [
                    sys.executable,
                    "scripts/generate_evalplus_jsonl.py",
                    "--model_path",
                    str(model_path),
                    "--dataset",
                    dataset,
                    "--output_jsonl",
                    str(jsonl),
                    "--seed",
                    str(seed),
                    "--threshold",
                    str(args.threshold),
                    "--batch_size",
                    str(args.batch_size),
                    "--max_new_tokens",
                    str(args.max_new_tokens),
                    "--bd_size",
                    str(args.bd_size),
                    "--small_block_size",
                    str(args.small_block_size),
                ]
                if args.use_carry:
                    gen_cmd.append("--use_carry")
                if args.use_block_cache_resolved:
                    gen_cmd.append("--use_block_cache")
                _run(gen_cmd, dry_run=args.dry_run)
            else:
                print(f"[run_evalplus_checkpoint] skip existing JSONL: {jsonl}")

            if args.force or not result_path.is_file():
                _run(
                    ["evalplus.evaluate", "--dataset", dataset, "--samples", str(jsonl)],
                    dry_run=args.dry_run,
                )
            else:
                print(f"[run_evalplus_checkpoint] skip existing EvalPlus result: {result_path}")

            if not args.dry_run:
                result_path = _result_path_for_jsonl(jsonl)
                loaded = _load_pass_at_1(result_path)
                prefix = f"evalplus/{dataset}/s{seed}"
                metrics[f"{prefix}/base_pass_at_1"] = loaded["base_pass_at_1"]
                metrics[f"{prefix}/plus_pass_at_1"] = loaded["plus_pass_at_1"]
                metrics[f"{prefix}/n_tasks"] = loaded["n_tasks"]

    summary_path = out_dir / f"{args.run_name}__{args.checkpoint_label}__summary.json"
    if not args.dry_run:
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
        print(f"Wrote {summary_path}")
        _maybe_log_wandb(args, metrics)


if __name__ == "__main__":
    main()
